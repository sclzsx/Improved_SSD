from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import torch.backends.cudnn as cudnn
from data import *
from layers.functions import Detect, PriorBox
from utils import nms
from utils.timer import Timer
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
import time

parser = argparse.ArgumentParser()
parser.add_argument('--version', default='SSD_VGG_Optim_FPN_RFB')
parser.add_argument('--trained_model', default='weights', type=str)
parser.add_argument('--size', default='300', type=str)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--cpu', default=False, type=bool)
parser.add_argument('--retest', default=False, type=bool)
parser.add_argument('--threshold', default=0.4, type=float)
args = parser.parse_args()

test_set = (DATASET.replace('_withneg', ''), 'trainval')
args.version = 'SSD_VGG_Optim_FPN_RFB'
dir_name = 'FocalLoss_BOTH_FPN_bz32'
args.trained_model = os.path.join(args.trained_model, DATASET, args.version, dir_name, 'SSD_VGG_Optim_FPN_RFB-FocalLoss_BOTH_FPN_bz32.pth')
source_videos_path = VOCroot + '/' + DATASET.replace('_withneg', '')
save_videos_dir = Path(args.trained_model).parent
source_images_root = VOCroot + '/' + DATASET.replace('_withneg', '') + '/test_images'

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

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

im_detect = Timer()


def detect(img, net, detector, cuda, transform, max_per_image=300, thresh=0.5):
    all_boxes = [[] for _ in range(num_classes)]
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    s = time.clock()
    out = net(x)  # forward pass
    e = time.clock()
    t = e - s
    fps = 1 / t
    print('FPS:', fps)
    boxes, scores = detector.forward(out, priors)
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    # scale each detection back up to the image
    c_dets = np.empty([0, 5], dtype=np.float32)
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            # c_dets = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = nms(c_dets, 0.45)
        c_dets = c_dets[keep, :]
        all_boxes[j] = c_dets
    return all_boxes


def load_net(img_dim):
    net = build_net('test', img_dim, num_classes)  # initialize detector
    state_dict = torch.load(args.trained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    # from ptflops import get_model_complexity_info
    # import time
    # import sys
    # with torch.no_grad():
    #     f, p = get_model_complexity_info(net, (3, 300, 300), as_strings=True, print_per_layer_stat=False, verbose=False)
    #     print('FLOPs:', f, 'Parms:', p)
    #     x = torch.randn(1, 3, 300, 300).cuda()
    #     s = time.clock()
    #     y = net(x)
    #     print(type(y), 1 / (time.clock() - s))
    # sys.exit()


    return net


def predict_videos(net, detector, part=False):
    video_w, video_h = 640, 480

    for vid_path in Path(source_videos_path).glob('*.avi'):
        dst_vid_path = str(save_videos_dir) + '/' + vid_path.name
        print('src:', str(vid_path))
        print('dst:', dst_vid_path)
        vw = cv2.VideoWriter(dst_vid_path, cv2.VideoWriter_fourcc(*'XVID'), 1, (video_w, video_h))
        cap = cv2.VideoCapture(str(vid_path))
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if part:
            stride = int(total_frame_num * 0.1)
        else:
            stride = 1
        for idx in tqdm(range(0, total_frame_num - 1, stride)):  # 抛弃最后一帧才能有效保存视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()

            all_boxes = detect(frame, net, detector, args.cuda, BaseTransform(img_dim, rgb_means, (2, 0, 1)),
                               top_k, thresh=args.threshold)
            for k in range(1, num_classes):
                if len(all_boxes[k]) > 0:
                    box_scores = all_boxes[k][:, 4]
                    box_locations = all_boxes[k][:, 0:4]
                    for i, box_location in enumerate(box_locations):
                        p1 = (int(box_location[0]), int(box_location[1]))
                        p2 = (int(box_location[2]), int(box_location[3]))
                        cv2.rectangle(frame, p1, p2, COLORS_BGR[k], 2)
                        title = "%s:%.2f" % (VOC_CLASSES[k], box_scores[i])
                        cv2.rectangle(frame, (p1[0] - 1, p1[1] - 1), (p2[0] + 1, p1[1] + 20), COLORS_BGR[k], -1)
                        # p3 = (max((p1[0] + p2[0]) // 2, 15), max((p1[1] + p2[1]) // 2, 15))
                        p3 = (p1[0] + 2, p1[1] + 15)
                        cv2.putText(frame, title, p3, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            frame = cv2.resize(frame, (video_w, video_h))
            if part:
                cv2.imwrite(os.path.join(save_videos_dir, "%d.jpg" % idx), frame)
            else:
                vw.write(frame)
        vw.release()


def predict_images(net, detector):
    source_images_dir = source_images_root + '/denoised'
    for img_path in Path(source_images_dir).glob('*.jpg'):
        dst_img_path = str(source_images_dir) + '_det' + '/' + img_path.name
        image = cv2.imread(str(img_path))
        all_boxes = detect(image, net, detector, args.cuda, BaseTransform(img_dim, rgb_means, (2, 0, 1)),
                           top_k, thresh=args.threshold)
        for k in range(1, num_classes):
            if len(all_boxes[k]) > 0:
                box_scores = all_boxes[k][:, 4]
                box_locations = all_boxes[k][:, 0:4]
                for i, box_location in enumerate(box_locations):
                    p1 = (int(box_location[0]), int(box_location[1]))
                    p2 = (int(box_location[2]), int(box_location[3]))
                    cv2.rectangle(image, p1, p2, COLORS_BGR[k], 2)
                    title = "%s:%.2f" % (VOC_CLASSES[k], box_scores[i])
                    cv2.rectangle(image, (p1[0] - 1, p1[1] - 1), (p2[0] + 1, p1[1] + 20), COLORS_BGR[k], -1)
                    # p3 = (max((p1[0] + p2[0]) // 2, 15), max((p1[1] + p2[1]) // 2, 15))
                    p3 = (p1[0] + 2, p1[1] + 15)
                    cv2.putText(image, title, p3, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.imwrite(dst_img_path, image)


if __name__ == '__main__':
    img_dim = (300, 512)[args.size == '512']
    net = load_net(img_dim)
    print(args.version, DATASET, VOC_CLASSES)
    detector = Detect(num_classes, 0, cfg)
    # predict_videos(net, detector, part=False)
    predict_images(net, detector)
