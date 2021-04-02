import torch
from torch.utils import data
import torch.nn as nn
import sys
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self,weight=None, reduction='none', gamma=1, eps=1e-7):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight,reduction=reduction) # reduction: mean

    def forward(self, input,target):
        logp = self.ce(input,target)
        p = torch.exp(-logp)
        loss = (1 - p)**self.gamma * logp
        return loss.mean()

def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[2]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0).unsqueeze(1)

    t = targets.unsqueeze(2)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)

    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


def sigmoid_focal_loss_cpu_v2(logits, targets, gamma, alpha):
    num_classes = logits.shape[2]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0).unsqueeze(1)

    t = targets.unsqueeze(2)
    p = torch.sigmoid(logits)

    posMask = (t == class_range)
    negMask = (t != class_range)
    posSigmoidOut = p[posMask]
    negSigmoidOut = p[negMask]

    term1 = ((1 - posSigmoidOut) ** gamma * torch.log(posSigmoidOut)).sum()
    term2 = (negSigmoidOut ** gamma * torch.log(1 - negSigmoidOut)).sum()

    if np.any(np.isnan(posSigmoidOut.cpu().detach().numpy())):
        print("posSigmoidOut has nan")
        sys.exit()
    if np.any(np.isnan(negSigmoidOut.cpu().detach().numpy())):
        print("negSigmoidOut has nan")
        sys.exit()

    return - term1 * alpha - term2 * (1 - alpha)

def sigmoid_focal_loss_cpu_v3(logits, targets, gamma, alpha):
    num_classes = logits.shape[2]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0).unsqueeze(1)

    t = targets.unsqueeze(2)

    pred = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1 - 1e-4)

    term1 = (1 - pred) ** gamma * torch.log(pred)
    term2 = pred ** gamma * torch.log(1 - pred)

    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)




class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        #device = logits.device
        #if logits.is_cuda:
        #    loss_func = sigmoid_focal_loss_cuda
        #else:
        #    loss_func = sigmoid_focal_loss_cpu
        loss_func = sigmoid_focal_loss_cpu_v3
        loss = loss_func(logits, targets, self.gamma, self.alpha)
        #print(loss)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

if __name__=='__main__':
    input = torch.randn((2,3,5))
    target = torch.empty((2,3),dtype=torch.long).random_(5)
    #weight = torch.tensor((0.25,0.75,0.75,0.75,0.75))
    #weight = None
    print(input,target)

    #loss = nn.CrossEntropyLoss()
    #focalloss1 = FocalLoss(gamma=2)
    #output_focalloss1 = focalloss1(input,target)

    focalloss2 = SigmoidFocalLoss(gamma=2,alpha=0.25)
    output_focalloss2 = focalloss2(input,target)
    #print(output_focalloss1)
    print(output_focalloss2)

