# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from collections import OrderedDict
from models.modules import *
import math

model = dict(
    type='Pelee',
    input_size=304,
    growth_rate=32,
    block_config=[3, 4, 8, 6],
    num_init_features=32,
    bottleneck_width=[1, 2, 4, 4],
    drop_rate=0.05,
)


class _DenseLayer(nn.Module):
    """docstring for _DenseLayer"""

    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ', inter_channel)

        self.branch1a = BasicConv(
            num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv(
            inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv(
            num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv(
            inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv(
            growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.branch1a(x)
        out1 = self.branch1b(out1)

        out2 = self.branch2a(x)
        out2 = self.branch2b(out2)
        out2 = self.branch2c(out2)

        out = torch.cat([x, out1, out2], dim=1)
        return out


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):

    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = BasicConv(
            num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = BasicConv(
            num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = BasicConv(
            num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv(
            2 * num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)

        return out


class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = BasicConv(in_channels, 128, kernel_size=1, bn=False)
        self.res1b = BasicConv(128, 128, kernel_size=3, padding=1, bn=False)
        self.res1c = BasicConv(128, 256, kernel_size=1, bn=False)

        self.res2a = BasicConv(in_channels, 256, kernel_size=1, bn=False)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out


class PeleeNet(nn.Module):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self):

        super(PeleeNet, self).__init__()

        self.growth_rate = 32
        self.block_config = [3, 4, 8, 6]
        self.num_init_features = 32
        self.bottleneck_width = [1, 2, 4, 4]
        self.drop_rate = 0.05

        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(3, self.num_init_features)),
        ]))

        if type(self.growth_rate) is list:
            growth_rates = self.growth_rate
            assert len(
                growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [self.growth_rate] * 4

        if type(self.bottleneck_width) is list:
            bottleneck_widths = self.bottleneck_width
            assert len(
                bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [self.bottleneck_width] * 4

        # Each denseblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=self.drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            self.features.add_module('transition%d' % (i + 1), BasicConv(
                num_features, num_features, kernel_size=1, stride=1, padding=0))

            if i != len(self.block_config) - 1:
                self.features.add_module('transition%d_pool' % (
                        i + 1), nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
                num_features = num_features


    def forward(self, x):
        sources = list()

        for k, feat in enumerate(self.features):
            x = feat(x)

            if k == 5 or k == 8 or k == len(self.features) - 1:
                sources += [x]

        return sources

    def init_model(self, pretained_model):


        def weights_init(m):
            '''
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(
                            m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
            '''
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.features.apply(weights_init)



if __name__ == '__main__':


    cfg = model
    input_size = 300
    net = PeleeNet()
    #print(net)

    '''
    # net.features.load_state_dict(torch.load('./peleenet.pth'))
    state_dict = torch.load('./weights/peleenet.pth')
    # print(state_dict.keys())
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[9:]] = v

    torch.save(new_state_dict, './weights/peleenet_new.pth')
    net.features.load_state_dict(new_state_dict)
    '''
    inputs = torch.randn(2, 3, 300, 300)
    out = net(inputs)

    for name,module in net.features.named_children():
        print(name)
        inputs = module(inputs)
        print(inputs.shape)
    # print(out.size())
