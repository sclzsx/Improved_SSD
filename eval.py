from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys

sys.path.append(os.getcwd())
from data import *
import argparse
import numpy as np
import pickle
from layers.functions import Detect, PriorBox
from tqdm import tqdm
from utils import nms
import json
from pathlib import Path

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str)
parser.add_argument('--trained_model', default='weights', type=str)
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--over_thresh', default=0.5, type=float, help='Cleanup and remove results files following eval')
parser.add_argument('--retest', default=False, type=bool, help='test cache results')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
args = parser.parse_args()

test_set = (DATASET.replace('_withneg', ''), 'test')
args.version = 'SSD_VGG_Optim_FPN_RFB'
dir_name = 'FocalLoss_BOTH_FPN_bz32'
train_model_dir = os.path.join(args.trained_model, DATASET, args.version, dir_name)
args.trained_model = os.path.join(train_model_dir, 'SSD_VGG_Optim_FPN_RFB-FocalLoss_BOTH_FPN_bz32.pth')

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


def del_file(path):
    if os.path.exists(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                del_file(path_file)


del_file(args.save_folder)

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        # obj_struct['difficult'] = int(obj.find('difficult').text)
        obj_struct['pose'] = '0'
        obj_struct['truncated'] = '0'
        obj_struct['difficult'] = '0'
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(data_dir, image_set, cls):
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(data_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(data_dir, all_boxes, dataset, set_type):
    for cls_ind, cls in enumerate(VOC_CLASSES):
        if cls_ind == 0:
            # print(cls_ind, cls)
            continue
        # get any class to store the result
        filename = get_voc_results_file_template(data_dir, set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s}*{:.3f}*{:.1f}*{:.1f}*{:.1f}*{:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir, data_dir, set_type, use_07=False, iou=0.5):
    cachedir = os.path.join(output_dir, 'annotations_cache')
    imgsetpath = os.path.join(data_dir, 'ImageSets', 'Main', test_set[1] + '.txt')
    annopath = os.path.join(data_dir, 'Annotations', '%s.xml')
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # print(cachedir)
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    # use_07_metric = True
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    for i, cls in enumerate(VOC_CLASSES):
        # if i == 0 or i == 1:
        if cls == '__background__' or cls == 'neg':
            # print(i, cls)
            continue
        filename = get_voc_results_file_template(output_dir, set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=iou, use_07_metric=use_07_metric)
        aps += [ap]

    return np.mean(aps), aps

    # return aps


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    cachefile = os.path.join(cachedir, 'annots.pkl.%s' % (test_set[0]))
    # read list of images
    # print(imagesetfile)
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # save the truth data as pickle,if the pickle in the file, just load it.
    # if not os.path.isfile(cachefile):
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split('*') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, dataset, net, detector, cuda, top_k, transform, im_size=300, thresh=0.05):
    # the len of pic
    num_images = len(dataset)
    # all detections are collected into:[21,4952,0]
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(VOC_CLASSES))]

    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        return all_boxes

    for i in tqdm(range(num_images)):
        with torch.no_grad():
            img = dataset.pull_image(i)
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

            out = net(x)

            boxes, scores = detector.forward(out, priors)

            boxes = boxes[0]
            scores = scores[0]

            boxes *= scale
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            # scale each detection back up to the image

            for j in range(1, num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

                keep = nms(c_dets, 0.45)
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    return all_boxes


def evaluate_detections(cache_dir, data_dir, box_list, dataset, eval_type='test'):
    result_dicts = []

    write_voc_results_file(cache_dir, box_list, dataset, eval_type)
    IouTh = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    mAPs = []
    for iou in IouTh:
        dict_tmp = dict()
        dict_tmp.setdefault('iou', round(iou, 2))

        mAP, aps = do_python_eval(cache_dir, data_dir, eval_type, use_07=False, iou=iou)
        dict_tmp.setdefault('mAP', round(float(mAP), 3))
        for i, ap in enumerate(aps):
            dict_tmp.setdefault(VOC_CLASSES[i + 2], round(ap, 3))

        mAPs.append(mAP)
        result_dicts.append(dict_tmp)
        print(dict_tmp)
    mAP_5095 = np.mean(mAPs)
    dict_tmp2 = dict()
    dict_tmp2.setdefault('mAP_50', round(float(mAPs[0]), 3))
    dict_tmp2.setdefault('mAP_5095', round(float(mAP_5095), 3))
    print(dict_tmp2)
    return result_dicts


# def evaluate(net):
#     num_classes = len(VOC_CLASSES)
#     dataset = VOCDetection(VOCroot, [('Shenzhen_VehiclePerson', 'test')])
#     if args.cuda:
#         net = net.cuda()
#         torch.backends.cudnn.benchmark = True
#     net.eval()
#
#     detector = Detect(num_classes, 0, cfg)
#
#     # evaluation
#     cache_path = args.save_folder
#     data_path = os.path.join(VOCroot, 'Shenzhen_VehiclePerson')
#
#     all_boxes = test_net(cache_path, dataset, net, detector, args.cuda, args.top_k, \
#                          BaseTransform(cfg['min_dim'], rgb_means, (2, 0, 1)), \
#                          im_size=cfg['min_dim'], thresh=args.confidence_threshold)
#
#     print('Evaluating detections')
#     result = evaluate_detections(cache_path, data_path, all_boxes, dataset, 'test')
#     return result


def load_net(img_dim, num_classes):
    try:
        net = build_net('test', img_dim, num_classes, neck_type='FPN')
    except:
        net = build_net('test', img_dim, num_classes)
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
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
    # print('Finished loading model!')
    # print(net)

    if args.cuda:
        net = net.cuda()
        torch.backends.cudnn.benchmark = True
    else:
        net = net.cpu()

    net.eval()
    return net


if __name__ == '__main__':
    dataset = VOCDetection(VOCroot, [test_set])
    print(args.trained_model)
    print(test_set, len(dataset), num_classes)
    print(VOC_CLASSES)

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                  CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    num_classes = len(VOC_CLASSES)
    # net = build_net('test', cfg['min_dim'], num_classes)  # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))

    net = load_net(cfg['min_dim'], num_classes)

    detector = Detect(num_classes, 0, cfg)

    # evaluation
    cache_path = args.save_folder
    data_path = os.path.join(VOCroot, test_set[0])

    all_boxes = test_net(cache_path, dataset, net, detector, args.cuda, args.top_k, \
                         BaseTransform(cfg['min_dim'], rgb_means, (2, 0, 1)), \
                         im_size=cfg['min_dim'], thresh=args.confidence_threshold)

    result_dict = evaluate_detections(cache_path, data_path, all_boxes, dataset, 'test')
    with open(train_model_dir + '/result_multi_10_denoised.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
