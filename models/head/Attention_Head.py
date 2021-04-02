import torch
import torch.nn as nn


def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)

class Attention(nn.Module):
    def __init__(self, c):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        f = self.conv1(x)   # [bs,c',h,w]
        g = self.conv2(x)   # [bs,c',h,w]
        h = self.conv3(x)   # [bs,c',h,w]

        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)    # [bs,N,c']
        g = hw_flattern(g)              # [bs,c',N]
        h = hw_flattern(h)              # [bs,c,N]
        h = torch.transpose(h, 1, 2)    # [bs,N,c]

        s = torch.matmul(f,g)           # [bs,N,N]
        beta = F.softmax(s, dim=-1)

        o = torch.matmul(beta,h)        # [bs,N,c]
        o = torch.transpose(o, 1, 2)
        o = o.view(x.shape)
        x = o + x
        return x




class Attention_Head(nn.Module):
    def __init__(self, num_classes):
        super(Attention_Head,self).__init__()

        self.loc_layers = list()
        self.conf_layers = list()
        self.atten_layers = list()
        self.anchors = 6

        # 38*38  512
        self.atten_layers += [Attention(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 19*19  512
        self.atten_layers += [Attention(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 10*10  256
        self.atten_layers += [Attention(128)]
        self.loc_layers += [nn.Conv2d(128, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(128, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 5*5  256
        self.atten_layers += [Attention(256)]
        self.loc_layers += [nn.Conv2d(256, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(256, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 3*3  256
        self.atten_layers += [Attention(256)]
        self.loc_layers += [nn.Conv2d(256, self.anchors * 4, kernel_size=1, padding=0)]
        self.conf_layers += [nn.Conv2d(256, self.anchors * num_classes, kernel_size=1, padding=0)]

        # 1*1  256
        self.atten_layers += [Attention(256)]
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


