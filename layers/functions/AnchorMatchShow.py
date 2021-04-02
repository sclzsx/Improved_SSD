from layers.modules.multibox_loss import match, match_ATSS
from data.voc0712 import VOCDetection
from data import *
from layers.functions import PriorBox
import torch
import cv2


# cfg = VOC_300
# cfg = VOC_300_3
cfg = VEHICLE_240
# train_sets = [('VehiclePersonV2_00', 'trainval'),
#               ('VOC2007', 'trainval'),
#               ('ShenzhenClean', 'trainval'),
#                #('negative','trainval'),
#                ]
train_sets = [('ShenzhenClean', 'trainval')]

dataset = VOCDetection(VOCroot, train_sets, preproc(
            cfg['min_dim'], rgb_means, p), AnnotationTransform())

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()

num_priors = (priors.size(0))

threshold = 0.5
variance = [0.1, 0.2]
anchor_match = 'NORMAL'
for index in range(len(dataset)):
    i = np.random.randint(0, len(dataset))
    img, targets = dataset[i]
    image = dataset.pull_image(i)
    h,w,_ = image.shape
    num = 1
    # match priors (default boxes) and ground truth boxes
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)

    for idx in range(num):
        truths = torch.from_numpy(targets[:, :-1].astype(np.float32))
        labels = torch.from_numpy(targets[:, -1])
        defaults = priors.data

        if anchor_match == 'ATSS':
            match_ATSS(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
        elif anchor_match == 'NORMAL':
            match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
        elif anchor_match == 'BOTH':
            match_ATSS(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
            # match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)


    pos = conf_t > 0
    pos_idx = pos.squeeze(0).unsqueeze(1).expand_as(priors)
    anchor_t = priors[pos_idx].view(-1, 4)
    #anchor_t = priors[(38*38//2*6):(38*38//2*6 + 6),:].view(-1, 4)
    anchor_t[:,0::2] *= w
    anchor_t[:,1::2] *= h

    for i in range(anchor_t.shape[0]):
        xmin = int(anchor_t[i,0] - anchor_t[i,2]/2)
        ymin = int(anchor_t[i,1] - anchor_t[i,3]/2)
        xmax = int(anchor_t[i,0] + anchor_t[i,2]/2)
        ymax = int(anchor_t[i,1] + anchor_t[i,3]/2)
        p1 = (xmin, ymin)
        p2 = (xmax, ymax)
        cv2.rectangle(image, p1, p2, (0, 255, 0),2)

    targets[:,0::2] *= w
    targets[:,1::2] *= h
    for i in range(targets.shape[0]):
        xmin = int(targets[i,0])
        ymin = int(targets[i,1])
        xmax = int(targets[i,2])
        ymax = int(targets[i,3])
        p1 = (xmin, ymin)
        p2 = (xmax, ymax)
        cv2.rectangle(image, p1, p2, (255, 0, 0),2)
    image = cv2.resize(image, (1280, 720))
    cv2.imshow("test", image)
    cv2.waitKey(0)