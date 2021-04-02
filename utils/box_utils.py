import torch
import torch.nn as nn
import math
import numpy as np
from data.config import VOC_300, VEHICLE_240
from itertools import product as product
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

INF = 100000000

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def outer_diagonal(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.max(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.min(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    outer = torch.clamp((max_xy - min_xy), min=0)
    outer_diag = (outer[:, :, 0] **2) + (outer[:, :, 1] **2)
    return outer_diag

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def jaccard_diou(box_a, box_b):
    # print("box_a:",box_a)
    # print("box_b:",box_b)
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    iou = inter / union
    # print("iou:",iou.t())
    outer_diag_distances = outer_diagonal(box_a, box_b)
    # print("outer_diag_distances: ",outer_diag_distances.t())

    # center distance
    gt_cx = (box_a[:, 2] + box_a[:, 0]) / 2.0
    gt_cy = (box_a[:, 3] + box_a[:, 1]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)
    # print("gt_points: ",gt_points)
    # print("truths:",truths)
    # print("gt_points: ",gt_points)
    # anchors_per_im = point_form(priors)
    anchors_cx_per_im = (box_b[:, 2] + box_b[:, 0]) / 2.0
    anchors_cy_per_im = (box_b[:, 3] + box_b[:, 1]) / 2.0
    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)
    # print("anchor_points: ",anchor_points)
    # print("anchor_points: ",anchor_points)
    center_distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1)
    # print("center_distances: ",center_distances)

    # dious = iou.t()
    # dious = iou.t() - (center_distances) / (outer_diag_distances.t())
    # dious = - (center_distances) / (outer_diag_distances.t())
    # dious = - (center_distances)
    dious =  (center_distances) / iou.t()
    # print("dious:",dious)
    dious = torch.clamp(dious,min=-1.0,max = 1.0)

    return dious

def matrix_iou(a,b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    conf[conf == 1] = 0
    #print(conf.sum())
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def anchor_level(VOC_300):
    feature_maps = VOC_300['feature_maps']
    aspect_ratios = VOC_300['aspect_ratios']
    anchor_level = []
    for k, f in enumerate(feature_maps):
        anchor_number = 0
        for i, j in product(range(f), repeat=2):
            anchor_number +=2
            # rest of aspect ratios
            for ar in aspect_ratios[k]:
                anchor_number +=2
        anchor_level.append(anchor_number)
    return anchor_level

def centerinbox(box_a, box_b):
    """compute whether the anchor(box_b) center is in the GT_Boxes(box_a).
    Return:
        isinbox: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    # print("box_a: ",box_a)
    # print("box_b: ",box_b)
    gt_minx = box_a[:, 0]
    gt_maxx = box_a[:, 2]
    gt_miny = box_a[:, 1]
    gt_maxy = box_a[:, 3]

    anchors_cx_per_im = (box_b[:, 2] + box_b[:, 0]) / 2.0
    anchors_cy_per_im = (box_b[:, 3] + box_b[:, 1]) / 2.0

    is_smaller_maxx = gt_maxx.unsqueeze(1).expand_as(inter) > anchors_cx_per_im.unsqueeze(0).expand_as(inter)

    is_larger_minx = gt_minx.unsqueeze(1).expand_as(inter) < anchors_cx_per_im.unsqueeze(0).expand_as(inter)

    is_smaller_maxy = gt_maxy.unsqueeze(1).expand_as(inter) > anchors_cy_per_im.unsqueeze(0).expand_as(inter)

    is_larger_miny = gt_miny.unsqueeze(1).expand_as(inter) < anchors_cy_per_im.unsqueeze(0).expand_as(inter)

    is_in_box = is_smaller_maxx * is_larger_minx * is_smaller_maxy * is_larger_miny

    return is_in_box


def match_ATSS(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    isinbox = centerinbox(truths, point_form(priors))
    in_box = False
    match_dis = "diou"

    if (match_dis == "l2"):
        # center distance
        gt_cx = (truths[:, 2] + truths[:, 0]) / 2.0
        gt_cy = (truths[:, 3] + truths[:, 1]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        anchors_per_im = point_form(priors)
        anchors_cx_per_im = (anchors_per_im[:, 2] + anchors_per_im[:, 0]) / 2.0
        anchors_cy_per_im = (anchors_per_im[:, 3] + anchors_per_im[:, 1]) / 2.0
        anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)
        distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
    elif(match_dis == "diou"):
        distances = jaccard_diou(truths, point_form(priors))
    # distances 的shape： [11640, GT]
    # ATSS implement
    center_dis_value = []
    center_dis_idx = []
    K = VEHICLE_240['anchor_level_k']
    # print('K',K)
    # prior_idx 每一层的anchor数目 [8664, 2166, 600, 150, 54, 6]
    prior_idx = anchor_level(VEHICLE_240)

    # print('prior_idx',prior_idx)
    star_idx = 0
    candidate_idxs = []
    for i in range(len(prior_idx)):
        end_idx = star_idx + prior_idx[i]
        distances_per_level = distances[star_idx:end_idx, :]
        topk = min(K[i], prior_idx[i])
        # topk_idxs_per_level 的shape[9, GT],每一列为与该GT度量最小的9个anchor序号
        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
        # print('topk_idxs_per_level + star_idx',topk_idxs_per_level + star_idx)
        candidate_idxs.append(topk_idxs_per_level + star_idx)
        star_idx = end_idx
    # candidate_idxs shape: [51, GT]
    candidate_idxs = torch.cat(candidate_idxs, dim=0)
    candidate_ious = overlaps.t()[candidate_idxs, torch.arange(truths.shape[0])]
    iou_mean_per_gt = candidate_ious.mean(0)
    iou_std_per_gt = candidate_ious.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

    anchor_num = priors.shape[0]
    for num_truth in range(truths.shape[0]):
        candidate_idxs[:, num_truth] += num_truth * anchor_num

    candidate_idxs = candidate_idxs.view(-1)

    ious_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
    index = candidate_idxs[is_pos.view(-1)]
    ious_inf[index] = overlaps.contiguous().view(-1)[index]

    ious_inf = ious_inf.view(truths.shape[0], -1).t()
    if (in_box):
        ious_inf[isinbox.t() == False] = -INF
    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

    conf = labels[anchors_to_gt_indexs]
    conf[anchors_to_gt_values == -INF] = 0
    # print('conf.shape',conf.shape)
    conf[conf == 1] = 0
    #print(conf.sum())
    matches = truths[anchors_to_gt_indexs]

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def GIOU_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    #loc = encode(matches, priors, variances)
    loc_t[idx] = matches    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_multi(matched, priors, offsets, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] - offsets[:,:2]
    # encode variance
    #g_cxcy /= (variances[0] * priors[:, 2:])
    g_cxcy.div_(variances[0] * offsets[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_multi(loc, priors, offsets, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + offsets[:,:2]+ loc[:, :2] * variances[0] * offsets[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count



def generalized_iou(Box_p,Box_gt):
    """
    Input:
        Box_p : 模型预测得到的物体的坐标信息,格式为(n,4)(x1,y,x2,y2),且
        Box_gt: 标注的物体坐标信息,格式为(n,4)(x1,y1,x2,y2)
    Output:
        loss_giou: 平均iou loss
    """
    assert Box_p.shape == Box_gt.shape
    # 转换数据格式
    Box_p = Box_p.float()
    Box_gt = Box_gt.float()
    # 确保格式为 x2>x1,y2>y1
    xp_1 = torch.min(Box_p[:,0],Box_p[:,2]).reshape(-1,1)
    xp_2 = torch.max(Box_p[:,0],Box_p[:,2]).reshape(-1,1)
    yp_1 = torch.min(Box_p[:,1],Box_p[:,3]).reshape(-1,1)
    yp_2 = torch.max(Box_p[:,1],Box_p[:,3]).reshape(-1,1)
    Box_p = torch.cat([xp_1,yp_1,xp_2,yp_2],1)
    # 计算预测框的面积
    box_p_area =  (Box_p[:,2]  - Box_p[:,0])  * (Box_p[:,3]  - Box_p[:,1])
    # 计算标签的面积
    box_gt_area = (Box_gt[:,2] - Box_gt[:,0]) * (Box_gt[:,3] - Box_gt[:,1])
    # 计算预测框与标签框之间的交集
    xI_1 = torch.max(Box_p[:,0],Box_gt[:,0])
    xI_2 = torch.min(Box_p[:,2],Box_gt[:,2])
    yI_1 = torch.max(Box_p[:,1],Box_gt[:,1])
    yI_2 = torch.min(Box_p[:,3],Box_gt[:,3])
    # 交集
    intersection =(yI_2 - yI_1) * (xI_2 - xI_1)
    #intersection = torch.max((yI_2 - yI_1),0) * torch.max((xI_2 - xI_1),0)
    # 计算得到最小封闭图形 C
    xC_1 = torch.min(Box_p[:,0],Box_gt[:,0])
    xC_2 = torch.max(Box_p[:,2],Box_gt[:,2])
    yC_1 = torch.min(Box_p[:,1],Box_gt[:,1])
    yC_2 = torch.max(Box_p[:,3],Box_gt[:,3])
    # 计算最小封闭图形C的面积
    c_area = (xC_2 - xC_1) * (yC_2 - yC_1)
    union = box_p_area + box_gt_area - intersection
    iou = intersection / union
    # GIoU
    giou = iou - (c_area - union) / c_area
    # GIoU loss
    loss_giou = 1 - giou

    #return loss_giou.mean()
    return loss_giou.sum()


def bbox_overlaps_diou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = ( outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious

def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious

def bbox_overlaps_giou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
    Returns:
        gious(ndarray): shape (n, k)
    """


    #bboxes1 = torch.FloatTensor(bboxes1)
    #bboxes2 = torch.FloatTensor(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    #print(ious)
    if exchange:
        ious = ious.T
    return ious



# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    #print(boxes)
    return boxes

def iou_loss(Box_p,Box_gt):
    """
    Input:
        Box_p : 模型预测得到的物体的坐标信息,格式为(n,4)(x1,y,x2,y2),且
        Box_gt: 标注的物体坐标信息,格式为(n,4)(x1,y1,x2,y2)
    Output:
        loss_giou: 平均iou loss
    """
    assert Box_p.shape == Box_gt.shape
    # 转换数据格式
    Box_p = Box_p.float()
    Box_gt = Box_gt.float()
    # 确保格式为 x2>x1,y2>y1
    xp_1 = torch.min(Box_p[:,0],Box_p[:,2]).reshape(-1,1)
    xp_2 = torch.max(Box_p[:,0],Box_p[:,2]).reshape(-1,1)
    yp_1 = torch.min(Box_p[:,1],Box_p[:,3]).reshape(-1,1)
    yp_2 = torch.max(Box_p[:,1],Box_p[:,3]).reshape(-1,1)
    Box_p = torch.cat([xp_1,yp_1,xp_2,yp_2],1)
    # 计算预测框的面积
    box_p_area =  (Box_p[:,2]  - Box_p[:,0])  * (Box_p[:,3]  - Box_p[:,1])
    # 计算标签的面积
    box_gt_area = (Box_gt[:,2] - Box_gt[:,0]) * (Box_gt[:,3] - Box_gt[:,1])
    # 计算预测框与标签框之间的交集
    xI_1 = torch.max(Box_p[:,0],Box_gt[:,0])
    xI_2 = torch.min(Box_p[:,2],Box_gt[:,2])
    yI_1 = torch.max(Box_p[:,1],Box_gt[:,1])
    yI_2 = torch.min(Box_p[:,3],Box_gt[:,3])
    # 交集
    intersection =(yI_2 - yI_1) * (xI_2 - xI_1)
    #intersection = torch.max((yI_2 - yI_1),0) * torch.max((xI_2 - xI_1),0)
    # 计算得到最小封闭图形 C
    xC_1 = torch.min(Box_p[:,0],Box_gt[:,0])
    xC_2 = torch.max(Box_p[:,2],Box_gt[:,2])
    yC_1 = torch.min(Box_p[:,1],Box_gt[:,1])
    yC_2 = torch.max(Box_p[:,3],Box_gt[:,3])
    # 计算最小封闭图形C的面积
    c_area = (xC_2 - xC_1) * (yC_2 - yC_1)
    union = box_p_area + box_gt_area - intersection
    iou = intersection / union
    # GIoU
    #giou = iou - (c_area - union) / c_area
    # GIoU loss
    #loss_giou = 1 - giou

    #return loss_giou.mean()

    #IoU loss
    loss_iou = - torch.log(iou)
    return loss_iou.mean()

if __name__ == "__main__":
    box_p = torch.tensor([[125,456,321,647],
                          [25,321,216,645],
                          [111,195,341,679],
                          [30,134,105,371]])
    box_gt = torch.tensor([[132,407,301,667],
                           [29,322,234,664],
                           [109,201,315,680],
                           [41,140,115,384]])
    giou_loss =  generalized_iou(box_p,box_gt)
    print(giou_loss)



