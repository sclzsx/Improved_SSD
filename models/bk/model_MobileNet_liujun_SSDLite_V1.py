from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_bn(inp,outp,kernel_size=3,padding=1,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,outp,kernel_size,stride,padding,bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU(inplace=True),
    )

def conv_bn_NoRelu(inp,outp,kernel_size=3,padding=1,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,outp,kernel_size,stride,padding,bias=False),
        nn.BatchNorm2d(outp),
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

def SeperableConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels,kernel_size=kernel_size,stride=stride,groups=in_channels,padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=1)
    )
class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = conv_bn(3, 32,stride=2)
        self.conv1_2 = conv_dw(32, 64)

        self.conv2_1 = conv_dw(64, 128,stride=2)
        self.conv2_2 = conv_dw(128, 128)

        self.conv3_1 = conv_dw(128, 256,stride=2)
        self.conv3_2 = conv_dw(256, 256)
        self.conv3_3 = conv_dw(256, 512,stride=2)

        self.conv4_1 = conv_dw(512, 512)
        self.conv4_2 = conv_dw(512, 512)
        self.conv4_3 = conv_dw(512, 512)

        self.conv5_1 = conv_dw(512, 512)
        self.conv5_2 = conv_dw(512, 512)
        self.conv5_3 = conv_dw(512, 1024,stride=2)

        self.conv6 = conv_dw(1024, 1024)

        # Load pretrained layers
        self.load_pretrained_layers()
        # Initialize convolutions' parameters
        #self.init_conv2d()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = self.conv1_1(image)  # (N, 32, 150, 150)
        out = self.conv1_2(out)  # (N, 64, 150, 150)

        out = self.conv2_1(out) # (N, 128, 75, 75)
        out = self.conv2_2(out)  # (N, 128, 75, 75)

        out = self.conv3_1(out)  # (N, 256, 38, 38)
        out = self.conv3_2(out)  # (N, 256, 38, 38)
        out = self.conv3_3(out)  # (N, 512, 38, 38)

        out = self.conv4_1(out)  # (N, 512, 19, 19)
        out = self.conv4_2(out) # (N, 512, 19, 19)
        out = self.conv4_3(out)  # (N, 512, 19, 19)

        out = self.conv5_1(out)  # (N, 512, 19, 19)
        out = self.conv5_2(out)  # (N, 512, 19, 19)

        conv4_3_feats = out  # (N, 512, 19, 19)
        #print('conv4_3_feats')
        #print(conv4_3_feats.cpu().data.numpy()[0, 0:10, 0,0])
        out = self.conv5_3(out)  # (N, 1024, 10, 10)

        out = self.conv6(out)  # (N, 1024, 10, 10)

        conv7_feats = out
        #print('conv7_feats')
        #print(conv7_feats.cpu().data.numpy()[0, 0:10, 0,0])

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we CONVERT fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in nets.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_model = torch.load('mobilenet_sgd_rmsprop_69.526.tar')
        pretrained_state_dict = pretrained_model['state_dict']
        pretrained_param_names = list(pretrained_state_dict.keys())

        print(param_names)
        # Transfer conv. parameters from pretrained model to current model
        j = 0
        for i, param in enumerate(param_names):  # excluding conv6 and conv7 parameters
            if param.split('.')[-1] == "num_batches_tracked":
                continue
            print(param,pretrained_param_names[j])
            state_dict[param] = pretrained_state_dict[pretrained_param_names[j]]
            j += 1

        '''
        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding
        '''
        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        '''
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        '''

        self.conv8_1 = nn.Conv2d(1024,256,kernel_size=1)
        self.conv8_2 = SeperableConv2d(256,512,stride=2)

        self.conv9_1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv9_2 = SeperableConv2d(128,256,stride=2)

        self.conv10_1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv10_2 = SeperableConv2d(128,256,stride=2)

        self.conv11_1 = nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.conv11_2 = SeperableConv2d(128,256,stride=2)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = self.conv8_1(conv7_feats)  # (N, 256, 10, 10)
        #print('conv8_1_feats')
        #print(out.cpu().data.numpy()[0, 0:10, 0,0])
        out = self.conv8_2(out)  # (N, 512, 5, 5)
        conv8_2_feats = out  # (N, 512, 5, 5)
        #print('conv8_2_feats')
        #print(conv8_2_feats.cpu().data.numpy()[0, 0:10, 0,0])


        out = self.conv9_1(out)  # (N, 128, 5, 5)
        out = self.conv9_2(out)  # (N, 256, 3, 3)
        conv9_2_feats = out  # (N, 256, 3, 3)
        #print('conv9_2_feats')
        #print(conv9_2_feats.cpu().data.numpy()[0, 0:10, 0,0])

        out = self.conv10_1(out)  # (N, 128, 3, 3)
        out = self.conv10_2(out)  # (N, 256, 2, 2)
        conv10_2_feats = out  # (N, 256, 2, 2)
        #print('conv10_2_feats')
        #print(conv10_2_feats.cpu().data.numpy()[0, 0:10, 0,0])

        out = self.conv11_1(out) # (N, 128, 2, 2)
        conv11_2_feats = self.conv11_2(out)  # (N, 256, 1, 1)
        #print('conv11_2_feats')
        #print(conv11_2_feats.cpu().data.numpy()[0, 0:10, 0,0])



        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in nets.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 6,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 6,
                   'conv11_2': 6}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = SeperableConv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = SeperableConv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = SeperableConv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = SeperableConv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = SeperableConv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = SeperableConv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = SeperableConv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = SeperableConv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = SeperableConv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = SeperableConv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map
        #print('l_conv4_3')
        #print(l_conv4_3.cpu().data.numpy()[0, 0:10, :])

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map
        #print('l_conv7')
        #print(l_conv7.cpu().data.numpy()[0, 0:10, :])

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)
        #print('l_conv8_2')
        #print(l_conv8_2.cpu().data.numpy()[0, 0:10, :])

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)
        #print('l_conv9_2')
        #print(l_conv9_2.cpu().data.numpy()[0, 0:10, :])

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)
        #print('l_conv10_2')
        #print(l_conv10_2.cpu().data.numpy()[0, 0:10, :])

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)
        #print('l_conv11_2')
        #print(l_conv11_2.cpu().data.numpy()[0, 0:4, :])

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        #print('c_conv4_3')
        #print(c_conv4_3.cpu().data.numpy()[0, 0:10, :])

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map
        #print('c_conv7')
        #print(c_conv7.cpu().data.numpy()[0, 0:10, :])

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)
        #print('c_conv8_2')
        #print(c_conv8_2.cpu().data.numpy()[0, 0:10, :])

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)
        #print('c_conv9_2')
        #print(c_conv9_2.cpu().data.numpy()[0, 0:10, :])

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)
        #print('c_conv10_2')
        #print(c_conv10_2.cpu().data.numpy()[0, 0:10, :])

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)
        #print('c_conv11_2')
        #print(c_conv11_2.cpu().data.numpy()[0, 0:4, :])

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        #print('locs')
        #print(locs.cpu().data.numpy()[0, 0:10, :])
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)
        #print('scores')
        #print(classes_scores.cpu().data.numpy()[0, 0:10, :])

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)


    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        #norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        #conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        #conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores



