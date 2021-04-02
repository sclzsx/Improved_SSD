import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import torch.nn.init as init
from models.modules import *


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, backbone, neck, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # SSD network
        self.base = nn.ModuleList(backbone)
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(512, 20)
        self.Norm = BasicRFB(128,128,stride = 1,scale=1.0)
        self.fpn = neck

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        fpn_sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(9):
            x = self.base[k](x)
            #print(x.shape)

        #s = self.L2Norm(x)
        #sources.append(x)
        fpn_sources.append(x)

        for k in range(9,15):
            x = self.base[k](x)
        fpn_sources.append(x)

        # apply vgg up to fc7
        for k in range(15, len(self.base)):
            x = self.base[k](x)
            #print(x.shape)
        #sources.append(x)
        fpn_sources.append(x)

        features = self.fpn(fpn_sources)

        features[0] = self.Norm(features[0])

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
base_channel = int(64 * 0.25)
def VGG():
    layers = []
    layers += [BasicConv(3, base_channel)]  
    layers += [BasicConv(base_channel, base_channel, kernel_size=3,stride=2, padding=1)] #150 * 150

    layers += [BasicConv(base_channel, base_channel * 2, kernel_size=3,stride=1, padding=1)]
    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1)] #75 * 75

    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1)] #38 * 38

    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)] # 19 * 19

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8,128,kernel_size=1,stride=1,padding=0)]
    layers += [BasicConv(128,128,kernel_size=3,stride=2, padding=1)] # 10*10

    return layers



class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Sequential(
            BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 256, kernel_size=3, stride=2, padding=1) # 5*5
        )

        self.P7 = nn.Sequential(
            BasicConv(256, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 256, kernel_size=3, stride=1, padding=0)  # 3 * 3
        )

        self.P8 = nn.Sequential(
            BasicConv(256, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 256, kernel_size=3, stride=1, padding=0)  # 1 * 1
        )

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7(P6_x)

        P8_x = self.P8(P7_x)


        return [P3_x, P4_x, P5_x, P6_x, P7_x, P8_x]


def build_head(cfg, num_classes):
    loc_layers = []
    conf_layers = []

    # 38*38  512
    loc_layers += [nn.Conv2d(base_channel * 8,cfg[0] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(base_channel * 8,cfg[0] * num_classes, kernel_size=1, padding=0)]

    # 19*19  512
    loc_layers += [nn.Conv2d(base_channel * 8,cfg[1] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(base_channel * 8,cfg[1] * num_classes, kernel_size=1, padding=0)]

    # 10*10  256
    loc_layers += [nn.Conv2d(base_channel * 8,cfg[2] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(base_channel * 8,cfg[2] * num_classes, kernel_size=1, padding=0)]

    # 5*5  256
    loc_layers += [nn.Conv2d(256,cfg[3] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(256,cfg[3] * num_classes, kernel_size=1, padding=0)]

     # 3*3  256
    loc_layers += [nn.Conv2d(256,cfg[4] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(256,cfg[4] * num_classes, kernel_size=1, padding=0)]

    # 1*1  256
    loc_layers += [nn.Conv2d(256,cfg[5] * 4, kernel_size=1, padding=0)]
    conf_layers +=[nn.Conv2d(256,cfg[5] * num_classes, kernel_size=1, padding=0)]

    return (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    backbone = VGG()
    neck = PyramidFeatures(128, 128, 128)
    head = build_head(mbox[str(size)], num_classes)
    return SSD(phase, backbone, neck, head, num_classes)



