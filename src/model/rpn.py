
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.util.util import IoU
from src.model.anchor_generate import generate_anchor_form, box_regression


class RPN(nn.Module):
    def __init__(self, feature_size, inter_channel, image_size, pos_thres=0.7, neg_thres=0.3):
        super().__init__()

        feature_channel, self.feature_H, self.feature_W = feature_size

        self.image_size = image_size
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres

        self._set_anchor()
        self.num_anchor = len(self.anchor)

        self.inter_conv = nn.Conv2d(feature_channel, inter_channel, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inter_channel, self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def _set_anchor(self):
        self.anchor = []
        for size in [128, 256, 512]:
            for ratio in [0.5, 1., 2.]:
                self.anchor.append([size, ratio])

    def forward(self, features):
        inter_feature = F.relu(self.inter_conv(features))
        cls_score = self.cls_conv(inter_feature)
        regression_variables = self.reg_conv(inter_feature)

        return cls_score, regression_variables

    def get_anchor_label(self, gt, regression_variables):
        '''
        B: batch size
        N: number of object
        k: number of anchor
        W: width of anchor feature map
        H: height of anchor feature map

        Args:
            gt (Dict) : {img, label, bbox}
                bbox (Tensor) : [B, N, 4]
            regression_variables : Tensor[B, 4*k, H, W]
               
        Returns:
            anchor_pos_label (Tensor) : [B, k, H, W] label True if anchor is selected as positve sample (highest, > Threshold)
            anchor_neg_label (Tensor) : [B, k, H, W] label True if anchor is selected as negative sample (< Threshold)
            anchor_bbox (Tensor) : [B, k, H, W, 4] # xywh
        '''
        # initialize
        bbox = gt['bbox']
        batch_size = bbox.size()[0]
        object_num = bbox.size()[1]
        BkHW = batch_size * self.num_anchor * self.feature_H * self.feature_W
        anchor_pos_label = torch.zeros((BkHW), dtype=torch.bool) # [B * k * H * W]
        anchor_neg_label = torch.zeros((BkHW), dtype=torch.bool) # [B * k * H * W]

        # set anchor        
        anchor_bbox = generate_anchor_form(self.anchor, (self.feature_H, self.feature_W), self.image_size) # [k, H, W, 4]
        anchor_bbox = anchor_bbox.repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]

        # box regression
        if regression_variables != None:
            moved_anchor_bbox = box_regression(anchor_bbox, regression_variables) # [B, k, H, W, 4]
        else:
            moved_anchor_bbox = anchor_bbox

        # expand bboxes for cross calculate
        anchor_bbox_cross = moved_anchor_bbox.repeat(object_num,1,1,1,1,1) # [B, k, H, W, 4] -> [N, B, k, H, W, 4]
        gt_bbox_cross = bbox.repeat(self.num_anchor, self.feature_H, self.feature_W, 1, 1, 1) # [B, N, 4] -> [k, H, W, B, N, 4]
        gt_bbox_cross = gt_bbox_cross.permute((4,3,0,1,2,5)) # [N, B, k, H, W, 4]

        # calculate IoU
        cross_IoU = IoU(gt_bbox_cross, anchor_bbox_cross).view((object_num, BkHW)) # [N, B * k * H * W]

        # label positive, negative anchor
        anchor_pos_label += torch.sum((cross_IoU > self.pos_thres), dim=0, dtype=torch.bool) # [B * k * H * W]
        anchor_neg_label += torch.logical_not(torch.sum((cross_IoU > self.neg_thres), dim=0, dtype=torch.bool)) # [B * k * H * W]

        # label highest anchor (and higher than negative threshold)
        _, highest_indices = torch.max(cross_IoU, dim=1) # [N]
        one_hot_label = F.one_hot(highest_indices, num_classes=BkHW) # [N, B * k * H * W]
        highest_label = torch.sum(one_hot_label, dim=0, dtype=torch.bool)# [B * k * H * W]
        anchor_pos_label += torch.logical_and(torch.logical_not(anchor_neg_label), highest_label)

        anchor_pos_label = anchor_pos_label.view((batch_size, self.num_anchor, self.feature_H, self.feature_W)) # [B, k, H, W]
        anchor_neg_label = anchor_neg_label.view((batch_size, self.num_anchor, self.feature_H, self.feature_W)) # [B, k, H, W]
        
        return anchor_pos_label, anchor_neg_label, moved_anchor_bbox

if __name__ == "__main__":
    xywh0 = torch.Tensor([1., 1., 1., 1.])
    xywh1 = torch.Tensor([0., 0., 3., 3.])

    print(xywh0)
    print(xywh1)
    print(IoU(xywh0, xywh1))