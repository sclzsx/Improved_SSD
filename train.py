from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import *
from layers.modules import MultiBoxLoss, FocalLossMultiBoxLoss, GIOUMultiBoxLoss
from layers.functions import PriorBox
from utils import *
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='SSD_Pelee',
                        help="SSD_VGG_RFB, SSD_Pelee, SSD_M2Det, SSD_DetNet, EfficientDet, SSD_VGG_Mobile_Little")
    parser.add_argument('--loss', default="OHEM", type=str, help='OHEM, GIOU, DIOU, CIOU, FocalLoss')
    parser.add_argument('--anchor', default='BOTH', type=str, help='BOTH, SSD, ATSS')
    parser.add_argument('--fpn_type', default='FPN', type=str, help='BIFPN, FPN, ABFPN, ACFPN')
    parser.add_argument('--size', default='300', help='300, 512')
    parser.add_argument('--save_folder', default='weights', type=str)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--lr', '--learning-rate', default=4e-3, type=float)
    parser.add_argument('--bz', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--resume_net', default=None, type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    return parser.parse_args()


def train(args):
    cfg = (VOC_300, VOC_512)[args.size == '512']
    if args.version == 'SSD_VGG_Mobile_Little':
        from models.SSD_VGG_Mobile_Little import build_net
        cfg = VEHICLE_240
    elif args.version == 'SSD_VGG_Optim_FPN_RFB':
        from models.SSD_VGG_Optim_FPN_RFB import build_net
    elif args.version == 'SSD_ResNet_FPN':
        from models.SSD_ResNet_FPN import build_net
    elif args.version == 'SSD_HRNet':
        from models.SSD_HRNet import build_net
    elif args.version == 'EfficientDet':
        from models.EfficientDet import build_net
    elif args.version == 'SSD_DetNet':
        from models.SSD_DetNet import build_net
        cfg = DetNet_300
    elif args.version == 'SSD_M2Det':
        from models.SSD_M2Det import build_net
        cfg = M2Det_320
    elif args.version == 'SSD_Pelee':
        from models.SSD_Pelee import build_net
    else:
        args.version = 'SSD_VGG_RFB'
        from models.SSD_VGG_RFB import build_net

    if args.loss == "OHEM":
        criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
    elif args.loss == "GIOU":
        criterion = GIOUMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
    elif args.loss == "DIOU":
        criterion = GIOUMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, loss_name='Diou')
    elif args.loss == "CIOU":
        criterion = GIOUMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, loss_name='Ciou')
    elif args.loss == "FocalLoss":
        criterion = FocalLossMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.anchor)

    if 'withneg' in DATASET:
        train_sets = [(DATASET.replace('_withneg', ''), 'trainval_withneg'), ]
    else:
        train_sets = [(DATASET.replace('_withneg', ''), 'trainval'), ]

    if args.resume_epoch == 0:
        args.save_folder = os.path.join(args.save_folder, DATASET, args.version,
                                        args.loss + '_' + args.anchor + '_' + args.fpn_type + '_bz' + str(args.bz))
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    else:
        args.save_folder = Path(args.resume_net).parent

    try:
        net = build_net('train', cfg['min_dim'], num_classes, args.fpn_type)
    except:
        net = build_net('train', cfg['min_dim'], num_classes)

    print(args.save_folder)
    try:
        flops, params = get_model_complexity_info(net, (cfg['min_dim'], cfg['min_dim']), print_per_layer_stat=False)
        print('FLOPs:', flops, 'Params:', params)
    except:
        pass

    init_net(net, args.resume_net)  # init the network with pretrained weights or resumed weights

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=momentum, weight_decay=weight_decay)

    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()

    dataset = VOCDetection(VOCroot, train_sets, preproc(cfg['min_dim'], rgb_means, p), AnnotationTransform())
    len_dataset = len(dataset)
    epoch_size = len_dataset // args.bz
    max_iter = args.max_epoch * epoch_size
    print(train_sets, 'len_dataset:', len_dataset, 'max_iter:', max_iter)

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues = stepvalues_VOC
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    if start_iter > stepvalues[0] and start_iter < stepvalues[1]:
        step_index = 1
    elif start_iter > stepvalues[1] and start_iter < stepvalues[2]:
        step_index = 2
    elif start_iter > stepvalues[2]:
        step_index = 3

    net.train()
    writer = SummaryWriter(args.save_folder)
    loc_loss = 0
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, args.bz,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate, pin_memory=True))
            loc_loss = 0
            conf_loss = 0
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(args.save_folder, str(epoch) + '.pth'))
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(args.lr, optimizer, gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)
        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) +
                  '||EpochIter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                  '||Totel iter ' + repr(iteration) +
                  '||L: %.4f C: %.4f' % (loss_l.item(), loss_c.item()) +
                  '||LR: %.8f' % (lr))
            writer.add_scalar('Train/total_loss', (loss_l.item() + loss_c.item()), iteration)
            writer.add_scalar('Train/loc_loss', loss_l.item(), iteration)
            writer.add_scalar('Train/conf_loss', loss_c.item(), iteration)
            writer.add_scalar('Train/lr', lr, iteration)

    torch.save(net.state_dict(), os.path.join(args.save_folder, str(args.max_epoch) + '.pth'))


def adjust_learning_rate(args_lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    if epoch < 6:
        lr = 1e-6 + (args_lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = get_args()
    # losses = ['OHEM, GIOU, DIOU, CIOU, FocalLoss']
    # anchors=['BOTH, SSD, ATSS']
    # fpn_types = ['BIFPN, FPN, ABFPN, ACFPN']
    train(args)
