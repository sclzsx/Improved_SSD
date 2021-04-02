import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import torch.nn.init as init

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def conv_bn(inp,outp,kernel_size=3,padding=1,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,outp,kernel_size,stride,padding,bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU(inplace=True),
    )

def SeperableConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels,kernel_size=kernel_size,stride=stride,groups=in_channels,padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=1)
    )

def conv_dw(inp,outp,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,inp,3,stride,1,groups=inp,bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp,outp,1,1,0,bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU(inplace=True),
    )

class CFEModule(nn.Module):
    def __init__(self,input,output):
        super(CFEModule,self).__init__()
        interchannels = input // 4
        self.branch0 = nn.Sequential(
            BasicConv(input,interchannels, kernel_size=1),
            #BasicConv(interchannels, interchannels,kernel_size=(7,1),groups=8,padding=(3,0)),
            #BasicConv(interchannels,interchannels,kernel_size=(1,7),groups=8,padding=(0,3)),
            conv_dw(interchannels,interchannels),
            BasicConv(interchannels,interchannels,kernel_size=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv(input, interchannels, kernel_size=1),
            #BasicConv(interchannels, interchannels, kernel_size=(1, 7), groups=8, padding=(3, 0)),
            #BasicConv(interchannels, interchannels, kernel_size=(7, 1), groups=8, padding=(0, 3)),
            conv_dw(interchannels,interchannels),
            BasicConv(interchannels, interchannels, kernel_size=1)
        )

        self.conv = BasicConv(2*interchannels,output,kernel_size=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0,x1),1)
        out = self.conv(out)

        return self.relu(x + out)



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

    def __init__(self, phase, size, base, extras, FFB,head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.FFBs = nn.ModuleList(FFB)

        self.CFE0 = CFEModule(256,256)
        self.CFE1 = CFEModule(256,256)
        self.CFE2 = CFEModule(256,256)
        self.CFE3 = CFEModule(256, 256)

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
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(9):
            x = self.base[k](x)    
        
        x_38 = x
        #print(x.shape)
        x = self.CFE0(x)
        for k in range(9, len(self.base)):
            x = self.base[k](x)

        x_19 = x

        x1 = self.FFBs[0](x_38)
        x2 = self.FFBs[1](x_19)
        x2 = self.FFBs[2](x2)
        xsum = x1 + x2
        xpre = self.CFE2(xsum)

        sources.append(xpre)
        #print(xpre.shape)

        x = self.CFE1(x)
        x = self.extras[0](x)
        x = self.extras[1](x)

        x_10 = x
        x1 = self.FFBs[3](x_19)
        x2 = self.FFBs[4](x_10)
        x2 = self.FFBs[5](x2)
        xsum = x1 + x2
        xpre = self.CFE3(xsum)

        sources.append(xpre)
        #print(xpre.shape)
        sources.append(x)
        #print(x.shape)
        x = self.extras[2](x)
        x = self.extras[3](x)
        sources.append(x)
        #print(x.shape)
        x = self.extras[4](x)
        x = self.extras[5](x)
        sources.append(x)
        #print(x.shape)
        x = self.extras[6](x)
        x = self.extras[7](x)
        sources.append(x)
        #print(x.shape)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
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
def MobileNetV1():
    layers = []
    layers += [conv_bn(3, 32, stride=2)]
    layers += [conv_dw(32, 64, 1)]
    layers += [conv_dw(64, 128, 2)]
    layers += [conv_dw(128, 128, 1)]
    layers += [conv_dw(128, 256, 2)]
    layers += [conv_dw(256, 256, 1)]
    layers += [conv_dw(256, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 256, 1)]
    layers += [conv_dw(256, 512, 2)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 256, 1)]

    return layers

base = {
    '320': [64, 'S', 128, 128, 'S', 256, 256, 'S',512, 512, 512, 512,512,512,'S',1024,1024],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

def FFB(input0,input1,input2,output):
    layers = []
    layers += [BasicConv(input0,output,kernel_size=1)]

    layers += [BasicConv(input1,output,kernel_size=1)]
    #layers += [nn.Upsample(scale_factor=2)]
    layers += [nn.ConvTranspose2d(output,output,kernel_size=2,stride=2,padding=0,bias=False)]

    layers += [BasicConv(input1,output,kernel_size=1)]

    layers += [BasicConv(input2,output,kernel_size=1)]
    #layers += [nn.Upsample(scale_factor=2)]
    layers += [nn.ConvTranspose2d(output,output,kernel_size=2,stride=2,padding=0,bias=False)]

    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [conv_bn(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]

            else:
                layers += [conv_bn(in_channels, v, kernel_size=(1, 3)[flag],padding=0)]
            flag = not flag
        in_channels = v
    layers += [conv_bn(128, 256, kernel_size=3, padding=0)]
    return layers

extras = {
    '320': [256, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
    #'320': [256, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}

def PredictionConvolutions(num_classes):
    n_boxes = {'conv3_3': 6,
               'conv4_3': 6,
               'conv6': 6,
               'conv7': 6,
               'conv8_2': 6,
               'conv9_2': 6,
               }

    loc_layers = []
    conf_layers = []

    for k in range(3):
        loc_layers += [SeperableConv2d(256,6 * 4)]
        conf_layers += [SeperableConv2d(256,6 * num_classes)]

    for k in range(3,6):
        loc_layers += [SeperableConv2d(256,6 * 4)]
        conf_layers += [SeperableConv2d(256,6 * num_classes)]

    return (loc_layers, conf_layers)

mbox = {
    '320': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 320 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    base_ = MobileNetV1()
    extras_ = add_extras(extras[str(size)], 256)
    head_ = PredictionConvolutions(num_classes)
    FFB_ = FFB(256,256,256,256)
    return SSD(phase, size, base_, extras_, FFB_,head_, num_classes)




if __name__=='__main__':
    '''
    net = build_net('train')
    print(net)
    image = torch.tensor(torch.ones((1, 3, 300, 300)))
    out = net(image)

    net = MobileNetV1()
    nnnet = nn.Sequential(*net)
    print(nnnet)
    '''

    deconv1 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=0,bias=False)

    input = Variable(torch.ones(1,1,2,2))
    out = deconv1(input)

    print(deconv1.weight)
    print(input)
    print(out)

    
