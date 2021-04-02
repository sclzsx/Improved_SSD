import torch
import torch.nn as nn
import torch.nn.init as init
from torch import distributed as dist

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def init_net(net, resume_net):
    if resume_net == None:

        net.apply(weights_init)

    else:
        # load resume network
        '''
        print('Loading resume network...')
        state_dict = torch.load(resume_net)
        model_dict = net.state_dict()
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k

            if name in model_dict:
                new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        '''
        net.apply(weights_init)

        pre_dict = torch.load(resume_net)
        model_dict = net.state_dict()

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for key, value in model_dict.items():
            new_state_dict[key] = value

            for p_key, p_value in pre_dict.items():
                head = p_key[:7]
                if head == 'module':
                    name = p_key[7:]
                else:
                    name = p_key
                
                if key == name:
                    new_state_dict[name] = p_value

        net.load_state_dict(new_state_dict)
                

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def adjust_learning_rate(optimizer, init_lr, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (init_lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = init_lr * (gamma ** (step_index))
    # lr = init_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        



