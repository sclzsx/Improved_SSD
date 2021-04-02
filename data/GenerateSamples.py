"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from data.config import VOC_CLASSES
from data.data_augment import _crop



# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


def creat_xml(xml_path,xml_name, bbox, label):

    f1 = open(os.path.join(xml_path, xml_name), 'w')
    f1.write("<annotation verified=\"no\">\n")
    f1.write("\t<folder>my_dataset</folder>\n")
    f1.write("\t<filename>" + xml_name + "</filename>\n")
    f1.write("\t<path>dataset</path>\n")
    f1.write("\t<source>\n")
    f1.write("\t<database>Unknown</database>\n")
    f1.write("\t</source>\n")
    f1.write("\t<size>\n")
    f1.write("\t\t<width>" + str(1920) + "</width>\n")
    f1.write("\t\t<height>" + str(1080) + "</height>\n")
    f1.write("\t\t<depth>3</depth>\n")
    f1.write("\t</size>\n")
    f1.write("\t<segmented>0</segmented>\n")

    for i in range(len(label)):
        name = VOC_CLASSES[int(label[i])]
        xmin = int(bbox[i,0])
        ymin = int(bbox[i,1])
        xmax = int(bbox[i,2])
        ymax = int(bbox[i,3])

        f1.write("\t<object>\n")
        f1.write("\t\t<name>" + name + "</name>\n")
        f1.write("\t\t<pose>Unspecified</pose>\n")
        f1.write("\t\t<truncated>0</truncated>\n")
        f1.write("\t\t<difficult>0</difficult>\n")
        f1.write("\t\t<bndbox>\n")
        f1.write("\t\t\t<xmin>" + str(xmin) + "</xmin>\n")
        f1.write("\t\t\t<ymin>" + str(ymin) + "</ymin>\n")
        f1.write("\t\t\t<xmax>" + str(xmax) + "</xmax>\n")
        f1.write("\t\t\t<ymax>" + str(ymax) + "</ymax>\n")
        f1.write("\t\t</bndbox>\n")
        f1.write("\t</object>\n")
    f1.write("</annotation>\n")
    f1.close()


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(name)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class ImagesGenerate(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            #rootpath = os.path.join(self.root, 'VOC' + year)
            rootpath = os.path.join(self.root, year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.dst_root_path = r"S:\liujun\DataSet\ADAS_VehiclePerson_SAMPLE\DataAugmentCrop"
        self.dst_anno_path = os.path.join(self.dst_root_path, 'Annotations')
        self.dst_img_path = os.path.join(self.dst_root_path, 'JPEGImages')

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        boxes = target[:,:-1].copy()
        labels = target[:,-1].copy()
        image_t, boxes, labels = _crop(img, boxes, labels)
        height, width, _ = image_t.shape

        # w = 1920
        # h = 1080
        #
        # left = random.randint(0, w - width)
        # top = random.randint(0, h - height)
        #
        # boxes_t = boxes.copy()
        # boxes_t[:, :2] += (left, top)
        # boxes_t[:, 2:] += (left, top)
        #
        # expand_image = np.empty(
        #     (h, w, 3),
        #     dtype=image_t.dtype)
        # expand_image[:, :] = rgb_means
        # expand_image[top:top + height, left:left + width] = image_t
        # image = expand_image

        image = cv2.resize(image_t, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        boxes[:, 0::2] = boxes[:, 0::2] / width * 1920
        boxes[:, 1::2] = boxes[:, 1::2] / height * 1080

        # for box in boxes:
        #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (25, 0,0),2)
        #
        # image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("1", image)
        # cv2.waitKey(0)

        cv2.imwrite("%s/crop_%06d.jpg"%(self.dst_img_path,index), image)

        creat_xml(self.dst_anno_path, 'crop_%06d.xml'%(index), boxes, labels)

        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

if __name__=="__main__":
    from data import *

    train_sets = [('VehiclePersonV2_00', 'trainval'),
                  ('VOC2007', 'trainval'),
                  ('ShenzhenClean', 'trainval'),
                  #('negative','trainval'),
                  #('NearBigCar', 'trainval'),
                #   ('VehiclePerson_lxn', 'trainval'),
                  ]

    dataset = ImagesGenerate(VOCroot, train_sets, None, AnnotationTransform())

    for i in range(len(dataset)):
        dataset[i]


