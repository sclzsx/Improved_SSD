import torch
import torch.nn as nn
from models.modules import *


def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)

class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = BasicConv(in_channels, 128, kernel_size=1, bn=False)
        self.res1b = BasicConv(128, 128, kernel_size=3, padding=1, bn=False)
        self.res1c = BasicConv(128, in_channels, kernel_size=1, bn=False)

        self.res2a = BasicConv(in_channels, in_channels, kernel_size=1, bn=False)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out




class Attention_Head(nn.Module):
    def __init__(self, num_classes):
        super(Attention_Head,self).__init__()

        self.loc_layers = list()
        self.conf_layers = list()
        self.atten_layers = list()
        self.anchors = 6

        # 38*38  512
        self.atten_layers += [ResBlock(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 19*19  512
        self.atten_layers += [ResBlock(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 10*10  256
        self.atten_layers += [ResBlock(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 5*5  256
        self.atten_layers += [ResBlock(256)]
        self.loc_layers += [nn.Conv2d(256, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(256, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 3*3  256
        self.atten_layers += [ResBlock(256)]
        self.loc_layers += [nn.Conv2d(256, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(256, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 1*1  256
        self.atten_layers += [ResBlock(256)]
        self.loc_layers += [nn.Conv2d(256, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(256, self.anchors * num_classes, kernel_size=1, padding=0)]

    def forward(self, features):
        loc = list()
        conf = list()

        for (x, a, l, c) in zip(features, self.atten_layers,self.loc_layers, self.conf_layers):
            x = a(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc,conf


