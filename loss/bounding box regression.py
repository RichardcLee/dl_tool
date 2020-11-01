# richard lee
# 2020.10.22
import torch
from math import pi
from torch import nn


# YOLO v4 bounding box regression loss
class BBoxIoULoss(nn.Module):
    def __init__(self, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False):
        super(BBoxIoULoss, self).__init__()
        self.eps = 1e-10
        self.GIoU = GIoU      # https://arxiv.org/pdf/1902.09630.pdf
        self.DIoU = DIoU      # https://arxiv.org/abs/1911.08287v1
        self.CIoU = CIoU      # https://arxiv.org/abs/1911.08287v1
        self.x1y1x2y2 =x1y1x2y2
        # self.anchor = torch.tensor([100, 100, 50, 50])  # todo!!! 如何实现

    def forward(self, box1, box2):
        """ calculate average x-iou loss for a batch.
        Args:
            box1: predict bounding boxes, shape is (batch_size, 4), each box is formed like [x,y,w,h] or [x1,y1,x2,y2]
            box2: groundtruth bounding boxes, shape is (batch_size, 4), each box is formed like [x,y,w,h] or [x1,y1,x2,y2]
        Return:
            iou or giou or diou or ciou loss.
        """
        # transform xywh to x1y1x2y2
        if self.x1y1x2y2 is False:
            box1_x1, box1_x2 = box1[:, 0] - box1[:, 2]/2, box1[:, 0] + box1[:, 2]/2
            box1_y1, box1_y2 = box1[:, 1] - box1[:, 3]/2, box1[:, 1] + box1[:, 3]/2

            box2_x1, box2_x2 = box2[:, 0] - box2[:, 2]/2, box2[:, 0] + box2[:, 2]/2
            box2_y1, box2_y2 = box2[:, 1] - box2[:, 3]/2, box2[:, 1] + box2[:, 3]/2

        else:
            box1_x1, box1_y1, box1_x2, box1_y2 = box1.t()[:]
            box2_x1, box2_y1, box2_x2, box2_y2 = box2.t()[:]

        # cal intersection area
        inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) *\
                (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

        # cal union area
        w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1
        w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1
        union = w1*h1 + self.eps + w2*h2 - inter

        # cal iou
        iou = inter / union

        if self.GIoU or self.DIoU or self.CIoU:
            # smallest enclosing box
            cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)    # width
            ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)    # height

            # cal Generalized IoU loss
            if self.GIoU:
                c = cw * ch + self.eps
                return (1 - iou + (c - union) / c).mean()

            # smallest enclosing box diagonal squared
            c2 = cw**2 + ch**2 + self.eps

            # center point distance squared
            p2 = ((box2_x1 + box2_x2) - (box1_x1 + box1_x2)) ** 2 / 4 +\
                 ((box2_y1 + box2_y2) - (box1_y1 + box1_y2)) ** 2 / 4

            # cal Distance IoU loss
            if self.DIoU:
                return (1 - iou + p2 / c2).mean()

            # cal Complete IoU loss
            if self.CIoU:
                # v measures the consistency of aspect ratio
                v = (4 / pi**2) * torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return (1 - iou + p2/c2 + alpha*v).mean()

        # cal IoU loss
        return (1 - iou).mean()