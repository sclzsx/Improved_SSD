import torch
import torch.nn as nn
from models.modules import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
import torch.utils.model_zoo as model_zoo

class DetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplannels, outplannels, stride=1, extra=True):
        super(DetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            BasicConv(inplannels, outplannels, kernel_size=1),
            BasicConv(outplannels, outplannels, kernel_size=3, stride=1, padding=2,dilation=2),
            nn.Conv2d(outplannels, outplannels, kernel_size=1),
            nn.BatchNorm2d(outplannels)
        )
        self.act = nn.ReLU(inplace=True)
        self.extra = extra
        if extra:
            self.extra = nn.Sequential(
                nn.Conv2d(inplannels, outplannels, kernel_size=1),
                nn.BatchNorm2d(outplannels)
            )

    def forward(self, x):
        x1 = self.bottleneck(x)
        if self.extra:
            out = x1 + self.extra(x)
        else:
            out = x1 + x

        out = self.act(out)
        return out

class DetNet18(nn.Module):

    def __init__(self, num_classes, block, detblock, layers,phase):

        self.inplanes = 64
        super(DetNet18, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # 38
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # 19
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)     # 10
        self.layer5 = self._make_layer(detblock, 256, layers[4], stride=1)  # 10
        self.layer6 = self._make_layer(detblock, 256, layers[5], stride=1)  # 10
        self.layer7 = self._make_layer(detblock, 256, layers[6], stride=1)  # 10

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)

        output = [x2, x3, x4, x5, x6, x7]

        return output
        #return x7

def DetNet(num_classes, pretrained=False, phase='train', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DetNet18(num_classes, BasicBlock, DetBottleneck, [2, 2, 2, 2, 2, 2, 2], phase, **kwargs)

    return model


if __name__=="__main__":
    x = torch.randn(4, 3, 300, 300)
    net = DetNet(3)

    for name, m in net.named_children():
        x = m(x)
        print(name, x.shape)