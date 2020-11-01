# richard lee
# 2020.10.22
import torch
from math import pi
from torch import nn
import torch.nn.functional as F


# focal loss
# https://www.jianshu.com/p/30043bcc90b6
class FocalLoss(nn.Module):
    """
        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            inputs = torch.sigmoid(inputs)  # 先激活
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1-pt)**self.gamma * bce

        if self.reduce:
            return torch.mean(focal)
        else:
            return focal
