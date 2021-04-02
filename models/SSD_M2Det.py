import torch
import torch.nn as nn
from torch.nn import init as init
from models.modules import *
import torch.nn.functional as F
from models.backbones.VGG import base_channel, VGG

class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 self.planes*self.num_levels // 16,
                                                 1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // 16,
                                                 self.planes*self.num_levels,
                                                 1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), BasicConv(self.in1, self.planes, 3, 2, 1))
        for i in range(self.scales -2):
            if not i == self.scales - 3:
                self.layers.add_module(
                    '{}'.format(len(self.layers)),
                    BasicConv(self.planes, self.planes, 3, 2, 1)
                )
            else:
                self.layers.add_module(
                    '{}'.format(len(self.layers)),
                    BasicConv(self.planes, self.planes, 3, 1, 0)
                )
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))

        self.latlayer = nn.Sequential()
        for i in range(self.scales -2):
            self.latlayer.add_module(
                '{}'.format(len(self.latlayer)),
                BasicConv(self.planes, self.planes, 3, 1, 1)
            )
        self.latlayer.add_module('{}'.format(len(self.latlayer)) ,BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales -1):
                smooth.append(
                    BasicConv(self.planes, self.planes, 1, 1, 0)
                )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _ ,_ ,H ,W = y.size()
        if fuse_type =='interp':
            return F.interpolate(x, size=(H ,W), mode='nearest') + y
        else:
            raise NotImplementedError
            # return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x ,y] ,1)
        conved_feat = [x]

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                self._upsample_add(
                    deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers ) - 1 -i])
                )
            )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                    self.smooth[i](deconved_feat[ i +1])
                )
            return smoothed_feat
        return deconved_feat

class M2Det(nn.Module):
    def __init__(self, phase, size, num_classes):
        '''
        M2Det: Multi-level Multi-scale single-shot object Detector
        '''
        super(M2Det ,self).__init__()
        self.phase = phase
        self.size = size


        self.backbone = 'vgg16'
        self.net_family = 'vgg' # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        self.base_out = [8,15] # [22,34] for vgg, [2,4] or [3,4] for res families
        self.planes = 128
        self.num_levels = 8
        self.num_scales = 6
        self.sfam = False
        self.smooth = True
        self.num_classes = num_classes

        self.construct_modules()



    def construct_modules(self ,):
        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(self,
                        'unet{}'.format( i +1),
                        TUM(first_level=True,
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=128))  # side channel isn't fixed.
            else:
                setattr(self,
                        'unet{}'.format( i +1),
                        TUM(first_level=False,
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=self.planes))
        # construct base features
        if 'vgg' in self.net_family:
            self.base = nn.ModuleList(VGG())
            shallow_in, shallow_out = 128 ,128
            deep_in, deep_out = 128 ,128
        self.reduce= BasicConv(shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(deep_in, deep_out, kernel_size=1, stride=1)
        # construct others
        if self.phase == 'test':
            self.softmax = nn.Softmax()
        #self.Norm = nn.BatchNorm2d(256 *8)
        self.leach = nn.ModuleList([BasicConv(
            deep_out +shallow_out,
            self.planes//2,
            kernel_size=(1 ,1) ,stride=(1 ,1)) ] *self.num_levels)

        # construct localization and recognition layers
        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes *self.num_levels,
                                  4 * 6, # 4 is coordinates, 6 is anchors for each pixels,
                                  3, 1, 1))
            conf_.append(nn.Conv2d(self.planes * self.num_levels,
                                   self.num_classes * 6,  # 6 is anchors for each pixels,
                                   3, 1, 1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)

    def forward(self ,x):
        loc ,conf = list() ,list()
        base_feats = list()
        if 'vgg' in self.net_family:
            for k in range(len(self.base)):
                x = self.base[k](x)

                if k in self.base_out:
                    base_feats.append(x)
                    #print(x.shape)

        elif 'res' in self.net_family:
            base_feats = self.base(x, self.base_out)

        # FFM1
        base_feature = torch.cat(
            (self.reduce(base_feats[0]), F.interpolate(self.up_reduce(base_feats[1]) ,scale_factor=2 ,mode='nearest'))
            ,1
        )

        # tum_outs is the multi-level multi-scale feature
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature), 'none')]


        for i in range(1 ,self.num_levels ,1):
            tum_outs.append(
                getattr(self, 'unet{}'.format( i +1))(
                    self.leach[i](base_feature), tum_outs[ i -1][-1]
                )
            )
        # concat with same scales
        sources = [torch.cat([_fx[ i -1] for _fx in tum_outs] ,1) for i in range(self.num_scales, 0, -1)]

        # forward_sfam
        if self.sfam:
            sources = self.sfam_module(sources)
        #sources[0] = self.Norm(sources[0])

        for (x ,l ,c) in zip(sources, self.loc, self.conf):
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

    def init_model(self, base_model_path):
        if self.backbone == 'vgg16':
            if isinstance(base_model_path, str):
                base_weights = torch.load(base_model_path)
                print('Loading base network...')
                self.base.load_state_dict(base_weights)
        elif 'res' in self.backbone:
            pass # pretrained seresnet models are initially loaded when defining them.

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        print('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        for i in range(self.num_levels):
            getattr(self ,'unet{}'.format( i +1)).apply(weights_init)
        self.reduce.apply(weights_init)
        self.up_reduce.apply(weights_init)
        self.leach.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_net(phase='train', size=320, num_classes=21):
    if not phase in ['test' ,'train']:
        raise ValueError("Error: Phase not recognized")

    if not size in [320]:
        raise NotImplementedError("Error: Sorry only M2Det320,M2Det512 M2Det704 or M2Det800 are supported!")

    return M2Det(phase, size, num_classes)



if __name__=="__main__":

    net = build_net(num_classes = 3)


    x = torch.randn(10,3,320,320)

    x = net(x)
    #print(net)
    #print(x.shape)

    from ptflops import get_model_complexity_info

    img_dim = 320
    flops, params = get_model_complexity_info(net, (img_dim, img_dim), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)