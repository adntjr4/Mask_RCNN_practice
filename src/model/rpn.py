
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.util.util import IoU, nms
from src.model.anchor_generate import generate_anchor_form, box_regression, calculate_regression_parameter


class RPN(nn.Module):
    def __init__(self, feature_size, inter_channel, image_size, threshold, reg_weight, nms_threshold):
        super().__init__()

        feature_channel, self.feature_H, self.feature_W = feature_size

        self.image_size = image_size
        self.pos_thres, self.neg_thres = threshold
        self.reg_weight = reg_weight
        self.nms_thres = nms_threshold

        self._set_anchor()
        self.num_anchor = len(self.anchor)

        # build layers
        self.inter_conv = nn.Conv2d(feature_channel, inter_channel, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inter_channel, 1*self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def _set_anchor(self):
        # anchor type definition
        self.anchor = []
        for size in [128, 256, 512]:
            for ratio in [0.5, 1., 2.]:
                self.anchor.append([size, ratio])

        # make default anchor
        self.register_buffer('default_anchor_bbox', generate_anchor_form(self.anchor, (self.feature_H, self.feature_W), self.image_size)) # [k, H, W, 4]
        #self.default_anchor_bbox = generate_anchor_form(self.anchor, (self.feature_H, self.feature_W), self.image_size).cuda() # [k, H, W, 4]

    def forward(self, features):
        inter_feature = F.relu(self.inter_conv(features))
        cls_score = torch.sigmoid(self.cls_conv(inter_feature)) # [B, 1*k, H, W]
        regression_variables = self.reg_conv(inter_feature)     # [B, 4*k, H, W]

        return {'cls_out': cls_score, 'reg_out': regression_variables}

    def get_anchor_label(self, gt, regression_variables):
        '''
        B: batch size
        N: number of object
        k: number of anchor
        W: width of anchor feature map
        H: height of anchor feature map
        P: number of positive anchor
        Args:
            gt (Dict) : {img, label, bbox}
                bbox (Tensor) : [B, N, 4]
            regression_variables (Tensor) : [B, 4*k, H, W]
               
        Returns:
            anchor_label (Dict) : {anchor_bbox, anchor_pos_label, anchor_neg_label, highest_gt}
                anchor_bbox (Tensor) : [B, k, H, W, 4] # xywh
                anchor_pos_label (Tensor) : [B, k, H, W] label True if anchor is selected as positve sample (highest, > Threshold)
                anchor_neg_label (Tensor) : [B, k, H, W] label True if anchor is selected as negative sample (< Threshold)
                highest_gt (Tensor) : [P, 2] (batch, object number)
                
        '''
        # initialize
        bbox = gt['bbox']
        batch_size = bbox.size()[0]
        object_num = bbox.size()[1]
        kHW = self.num_anchor * self.feature_H * self.feature_W
        anchor_pos_label = torch.zeros((batch_size, kHW), dtype=torch.bool).cuda() # [BkHW]
        anchor_neg_label = torch.zeros((batch_size, kHW), dtype=torch.bool).cuda() # [BkHW]

        # anchor_bbox expand as batch size
        anchor_bbox = self.default_anchor_bbox.repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]

        # box regression
        if regression_variables != None:
            moved_anchor_bbox = box_regression(anchor_bbox, regression_variables, self.reg_weight) # [B, k, H, W, 4]
        else:
            moved_anchor_bbox = anchor_bbox

        if object_num != 0:
            # expand bboxes for cross calculate
            anchor_bbox_cross = moved_anchor_bbox.repeat(object_num,1,1,1,1,1).permute(1,0,2,3,4,5) # [B, k, H, W, 4] -> [B, N, k, H, W, 4]
            gt_bbox_cross = bbox.repeat(self.num_anchor, self.feature_H, self.feature_W, 1, 1, 1) # [B, N, 4] -> [k, H, W, B, N, 4]
            gt_bbox_cross = gt_bbox_cross.permute((3,4,0,1,2,5)) # [B, N, k, H, W, 4]

            # calculate IoU
            cross_IoU = IoU(anchor_bbox_cross, gt_bbox_cross).view((batch_size, object_num, kHW)) # [B, N, kHW]
            
            # label highest anchor
            _, highest_indices = torch.max(cross_IoU, dim=2) # [B, N]
            one_hot_label = F.one_hot(highest_indices, num_classes=kHW) # [B, N, kHW]
            highest_label = torch.sum(one_hot_label, dim=1, dtype=torch.bool) # [B, kHW]
            anchor_pos_label += torch.logical_and(torch.logical_not(anchor_neg_label), highest_label)

            # label positive, negative anchor ()
            anchor_pos_label += torch.sum((cross_IoU > self.pos_thres), dim=1, dtype=torch.bool) # [B, kHW]
            anchor_neg_label += torch.logical_not(torch.sum((cross_IoU > self.neg_thres), dim=1, dtype=torch.bool)) # [B, kHW]

            anchor_pos_label = anchor_pos_label.view((batch_size, self.num_anchor, self.feature_H, self.feature_W)) # [B, k, H, W]
            anchor_neg_label = anchor_neg_label.view((batch_size, self.num_anchor, self.feature_H, self.feature_W)) # [B, k, H, W]

            # find hightest gt bbox of each anchor
            highest_gt_per_anchor = torch.max(cross_IoU, dim=1)[1].view(batch_size, self.num_anchor, self.feature_H, self.feature_W) # [B, k, H, W] (int:0~N-1)
            highest_gt_object = highest_gt_per_anchor[anchor_pos_label] # [P] (int:0~N-1)
            highest_gt_batch  = torch.arange(0, batch_size).cuda().repeat(self.num_anchor, self.feature_H, self.feature_W, 1).permute(3,0,1,2)[anchor_pos_label] # [P] (int:0~B-1)
            highest_gt = torch.stack([highest_gt_batch, highest_gt_object], dim=1)
        else:
            anchor_pos_label = anchor_pos_label.view((batch_size, self.num_anchor, self.feature_H, self.feature_W))
            anchor_neg_label = torch.logical_not(anchor_neg_label).view((batch_size, self.num_anchor, self.feature_H, self.feature_W))
            highest_gt = torch.zeros((0, 2), dtype=torch.int).cuda()

        # return dictionary
        return_dict = dict()
        return_dict['anchor_bbox'] = moved_anchor_bbox
        return_dict['anchor_pos_label'] = anchor_pos_label
        return_dict['anchor_neg_label'] = anchor_neg_label
        return_dict['highest_gt'] = highest_gt

        return return_dict

    def get_proposed_RoI(self, cls_score, regression_variables, threshold):
        '''
        R : number of proposed RoI
        Args:
            cls_score (Tensor) : [B, 1*k, H, W]
            regression_variables (Tensor) : [B, 4*k, H, W]
        Returns:
            RoI_bbox (Tensor) : [R, 4]
        '''
        batch_size = cls_score.size()[0]

        # anchor_bbox expand as batch size
        anchor_bbox = self.default_anchor_bbox.repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]
        
        # box regression
        if regression_variables != None:
            moved_anchor_bbox = box_regression(anchor_bbox, regression_variables, self.reg_weight) # [B, k, H, W, 4]
        else:
            moved_anchor_bbox = anchor_bbox

        RoI_bool = cls_score > threshold
        RoI_bbox = nms(moved_anchor_bbox[RoI_bool], cls_score[RoI_bool], self.nms_thres)

        return RoI_bbox

    def RPN_label_select(self, cls_out, anchor_label, sample_number):
        '''
        random sampling positive, negative anchor with limited number
        P : number of positive anchor
        N : number of negative anchor
        Args:
            cls_out : Tensor[B, k, H, W]
            anchor_label : {anchor_pos_label}
                anchor_pos_label : Tensor[B, k, H, W] (bool)
                anchor_neg_label : Tensor[B, k, H, W] (bool)
        Return:
            sampled_cls_out : Tensor[sample_number]
            label : Tensor[P+N] (front:ones, back:zeros)
        '''
        pos_label, neg_label = anchor_label['anchor_pos_label'], anchor_label['anchor_neg_label']

        B, k, H, W = cls_out.size()

        # get whole positive and negative class out
        pos_cls_out, neg_cls_out = cls_out[pos_label],           cls_out[neg_label]
        pos_num,     neg_num     = cls_out[pos_label].size()[0], cls_out[neg_label].size()[0]

        # random sampling
        pivot = int(sample_number/2)
        if pos_num <= pivot:
            sampled_pos_cls_out = pos_cls_out
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sample_number-pos_num]]
            sampled_pos_num = pos_num
            sampled_neg_num = sample_number-pos_num
        else:
            sampled_pos_cls_out = pos_cls_out[torch.randperm(pos_num)[:pivot]]
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sample_number-pivot]]
            sampled_pos_num = pivot
            sampled_neg_num = sample_number-pivot

        sampled_cls_out = torch.cat([sampled_pos_cls_out, sampled_neg_cls_out])

        label = torch.cat([torch.ones((sampled_pos_num)), torch.zeros((sampled_neg_num))]).cuda()

        return sampled_cls_out, label

    def RPN_cal_t_regression(self, reg_out, gt, anchor_label):
        '''
        P : number of positive anchor
        Args:
            reg_out (Tensor) : [B, 4*k, H, W]
            gt : {img, img_size, label, bbox}
                bbox (Tensor) : [B, N, 4]
            anchor_label : {anchor_bbox, anchor_pos_label, pos_indice}
                anchor_bbox (Tensor) : [B, k, H, W, 4]
                anchor_pos_label (Tensor) : [B, k, H, W] (bool)
                highest_gt (Tensor) : [P, 2] (batch, object number)
        Returns:
            predicted_t  : Tensor[P, 4]
            calculated_t : Tensor[P, 4]
        '''
        anchor_bbox, pos_label, highest_gt = anchor_label['anchor_bbox'], anchor_label['anchor_pos_label'], anchor_label['highest_gt']

        B, k4, H, W = reg_out.size()
        k = int(k4/4)

        # reshape reg_out for predicted_t
        predicted_t = reg_out.view(B, k, 4, H, W).permute(0,1,3,4,2)[pos_label] # [P, 4]

        # calculate box regression parameter
        pos_anchor_bbox = anchor_bbox[pos_label] # [P, 4] (xywh)
        pos_gt_bbox = torch.stack([gt['bbox'][batch_num][gt_num] for batch_num, gt_num in highest_gt]) # [P, 4] (xywh)

        calculated_t = calculate_regression_parameter(pos_anchor_bbox, pos_gt_bbox, self.reg_weight)

        return predicted_t, calculated_t


if __name__ == "__main__":
    xywh0 = torch.Tensor([1., 1., 1., 1.])
    xywh1 = torch.Tensor([0., 0., 3., 3.])

    print(xywh0)
    print(xywh1)
    print(IoU(xywh0, xywh1))