import torch
import torch.nn as nn
from models.modules import BasicConv




class Bottleneck(nn.Module):
    def __init__(self, c1, c2, short_cut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c1 * e)
        self.conv1 = BasicConv(c1, c_, kernel_size=1)
        self.conv2 = BasicConv(c_, c2, kernel_size=3, padding=1)
        self.add = short_cut and c1 == c2

    def forward(self, x):
        y1 = self.conv2(self.conv1(x))

        return y1 + x if self.add else y1


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, short_cut=True, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = BasicConv(c1, c_, kernel_size=1)
        self.bottleneck = nn.Sequential(*[Bottleneck(c_, c_, True, 1) for _ in range(n)])
        self.conv2 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)

        self.conv3 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2*c_)
        self.act1 = nn.ReLU(inplace=True)

        self.conv4 = BasicConv(2 * c_, c2, kernel_size=1)

    def forward(self, x):
        y1 = self.conv2(self.bottleneck(self.conv1(x)))
        y2 = self.conv3(x)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv4(self.act1(self.bn1(y)))

        return y

class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.conv1 = BasicConv(c1, c_, kernel_size=1)
        self.pool = nn.ModuleList([nn.MaxPool2d(i, stride=1, padding= i // 2) for i in k])
        self.conv2 = BasicConv(c_ * (len(k) + 1), c2, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        y = torch.cat([y]+[m(y) for m in self.pool], dim=1)
        y = self.conv2(y)

        return y

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.x = 5

    def forward(self, x):
        print(self.x)
        return self.x

class CSPNet(nn.Module):
    def __init__(self, wr=0.5, dr=0.33):
        super(CSPNet, self).__init__()
        base_c = int(64 * wr)
        self.conv1 = BasicConv(3, base_c, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv(base_c, base_c * 2, kernel_size=3, stride=2, padding=1)
        self.block1 = BottleneckCSP(base_c * 2, base_c * 2)
        self.conv3  = BasicConv(base_c * 2, base_c * 4, kernel_size=3, stride=2, padding=1)
        self.block2 = BottleneckCSP(base_c * 4, base_c * 4)
        self.conv4  = BasicConv(base_c * 4, base_c * 8, kernel_size=3, stride=2, padding=1)
        self.block3 = BottleneckCSP(base_c * 8, base_c * 8)
        self.conv5  = BasicConv(base_c * 8, base_c * 16, kernel_size=3, stride=2, padding=1)
        self.spp    = SPP(base_c * 16, base_c * 16)

    def forward(self, x):
        y1 = self.block2(self.conv3(self.block1(self.conv2(self.conv1(x)))))
        y2 = self.block3(self.conv4(y1))
        y3 = self.spp(self.conv5(y2))

        return [y1, y2, y3]

class PANet(nn.Module):
    def __init__(self, base_c):
        super(PANet, self).__init__()
        self.csp1 = BottleneckCSP(base_c * 16, base_c * 16)
        self.conv1 = BasicConv(base_c * 16, base_c * 8, kernel_size=1)
        self.trans1 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)

        self.csp2 = BottleneckCSP(base_c * 16, base_c * 8)
        self.conv2 = BasicConv(base_c * 8, base_c * 4, kernel_size=1)
        self.trans2 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)

        self.csp3 = BottleneckCSP(base_c * 8, base_c * 4)

        self.conv4 = BasicConv(base_c * 4, base_c * 4, kernel_size=3, stride=2, padding=1)
        self.csp4 = BottleneckCSP(base_c * 8, base_c * 8)

        self.conv5 = BasicConv(base_c * 8, base_c * 8, kernel_size=3, stride=2, padding=1)
        self.csp5 = BottleneckCSP(base_c * 16, base_c * 16)

    def forward(self, x1, x2, x3):
        y3 = self.conv1(self.csp1(x3))
        y2 = self.conv2(self.csp2(torch.cat([x2, self.trans1(y3)], dim=1)))
        y1 = torch.cat([x1, self.trans2(y2)], dim=1)

        p1 = self.csp3(y1)
        p2 = self.csp4(torch.cat([y2, self.conv4(p1)], dim=1))
        p3 = self.csp5(torch.cat([y3, self.conv5(p2)], dim=1))

        return [p1, p2, p3]

class yolov5(nn.Module):
    def __init__(self):
        super(yolov5, self).__init__()
        wr = 0.25
        base_c = int(64 * wr)
        self.backbone = CSPNet(wr)
        self.neck = PANet(base_c)

    def forward(self, x):
        y1, y2, y3 = self.backbone(x)

        p1, p2, p3 = self.neck(y1, y2, y3)

        return [p1, p2, p3]
if __name__=="__main__":
    x = torch.randn((2,3,640, 640))
    model = yolov5()

    p1, p2, p3 = model(x)
    print(p1.shape, p2.shape, p3.shape)

    from ptflops import get_model_complexity_info

    img_dim = 640
    flops, params = get_model_complexity_info(model, (img_dim, img_dim), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)

