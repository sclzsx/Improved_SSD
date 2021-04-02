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
from models.backbones.CSPResNet import yolov5


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

    def __init__(self, phase, backbone, neck, head, num_classes, img_dim=300):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # SSD network
        self.base = backbone
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(512, 20)
        # self.Norm = BasicRFB(128,128,stride = 1,scale=1.0)
        # self.fpn = neck
        self.extras = nn.ModuleList(add_extras(img_dim))

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
        # apply vgg up to conv4_3 relu
        features = self.base(x)
        # features = self.fpn(fpn_sources)
        # features[0] = self.Norm(features[0])

        x = features[-1]
        for k, v in enumerate(self.extras):
            x = v(x)
            #print(x.shape)
            if k > 0 and k % 4 == 0:
                features.append(x)

        # for f in features:
        #     print(f.shape)

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


def add_extras(img_dim=640):
    # Extra layers added to VGG for feature scaling
    layers = []

    if img_dim == 640:
        layers += [BasicRFB(256,128,stride = 1,scale=1.0)]

        layers += [BasicConv(128, 128,kernel_size=1,stride=1,padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128,kernel_size=3,stride=2, padding=1)] # 10 * 10

        layers += [BasicConv(128, 128,kernel_size=1,stride=1,padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128,kernel_size=3,stride=2, padding=1)] # 5 * 5

        layers += [BasicConv(128, 128,kernel_size=1,stride=1,padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 128,kernel_size=3,stride=2, padding=1)] # 3 * 3

    return layers

def build_head(cfg, num_classes):
    loc_layers = []
    conf_layers = []

    # 80
    loc_layers += [nn.Sequential(
                      BasicConv(64, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(64, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    # 40
    loc_layers += [nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    # 20
    loc_layers += [nn.Sequential(
                      BasicConv(256, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(256, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    # 10
    loc_layers += [nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    # 5
    loc_layers += [nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    # 3
    loc_layers += [nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * 4, kernel_size=1, padding=0))]
    conf_layers +=[nn.Sequential(
                      BasicConv(128, 128, kernel_size=1, stride=1, padding=0),
                      nn.Conv2d(128,cfg[0] * num_classes, kernel_size=1, padding=0))]

    return (loc_layers, conf_layers)

mbox = {
    '640': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
}


def build_net(phase, size=640, num_classes=21,neck_type='FPN'):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 640:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    backbone = yolov5()
    neck = None
    head = build_head(mbox[str(size)], num_classes)
    return SSD(phase, backbone, neck, head, num_classes, img_dim=size)


if __name__=='__main__':

    x = torch.rand(2,3,640,640)
    model = build_net('train', size=640)
    output = model(x)

    from ptflops import get_model_complexity_info

    img_dim = 640
    flops, params = get_model_complexity_info(model, (img_dim, img_dim), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)

