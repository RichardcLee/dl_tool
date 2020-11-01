# richard lee
# 2020.10.22
from torch import nn
import torch


# dice loss
class DiceLoss(nn.Module):
    def __init__(self, mode="average", logit=False):
        super(DiceLoss, self).__init__()
        self.eps = 1e-10   # 1.0
        self.mode = mode
        self.logit = logit

    def forward(self, predictions, targets):
        """
            calculate a batch's average dice loss or a batch(as a whole)'s dice loss
        """
        if not self.logit:
            predictions = torch.sigmoid(predictions)

        if self.mode == "average":  # batch中每个图像视为单独个体 最后求平均
            # score = tensor(0.)
            # for prediction, target in zip(predictions, targets):
            #     i_flat = prediction.view(-1)
            #     t_flat = target.view(-1)
            #
            #     intersection = (i_flat * t_flat).sum()
            #
            #     score += 1 - ((2. * intersection + self.eps) / (i_flat.sum() + t_flat.sum() + self.eps))
            #
            # score /= predictions.size()[0]
            # return score

            N = predictions.size(0)  # batch size or input numbers
            i_flat = predictions.view(N, -1)
            t_flat = targets.view(N, -1)
            intersection = (i_flat * t_flat).sum(dim=1)
            score = (1 - ((2. * intersection + self.eps) / (i_flat.sum(dim=1) + t_flat.sum(dim=1) + self.eps))).mean()

            return score

        elif self.mode == "whole":   # batch视为一个整体求dice
            i_flat = predictions.view(-1)
            t_flat = targets.view(-1)

            intersection = (i_flat * t_flat).sum()

            return 1 - ((2. * intersection + self.eps) / (i_flat.sum() + t_flat.sum() + self.eps))

        else:
            raise Exception("No such dice mode: %s" % self.mode)


# BCE dice
# TODO