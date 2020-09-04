
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.util.util import IoU, nms
from src.model.rpn.rpn_head import RPNHead
from src.model.anchor_generate import generate_anchor_form, box_regression, calculate_regression_parameter


class RPN(nn.Module):
    '''
    F: number of feature map
    B: batch size
    N: number of object
    k: number of anchor
    W: width of anchor feature map
    H: height of anchor feature map
    P: number of positive anchor
    '''

    def __init__(self, FPN_mode, feature_channel, feature_size, conf_RPN):
        super().__init__()

        self.rpn_head = RPNHead(FPN_mode, conf_RPN)

        self.FPN_mode = FPN_mode
        self.pos_thres = conf_RPN['positive_threshold']
        self.neg_thres = conf_RPN['negative_threshold']
        self.init_weight = conf_RPN['parameter_init_weight']
        self.reg_weight = conf_RPN['regression_weight']
        self.nms_thres = conf_RPN['nms_threshold']

        self.feature_size = feature_size
        self.image_size = conf_RPN['input_size']

        self.anchor = self.rpn_head.get_anchor_type()
        self.num_anchor = self.rpn_head.get_anchor_number()
        self._set_anchor()

        # build layers
        inter_channel = conf_RPN['intermediate_channel_number']
        self.inter_conv = nn.Conv2d(feature_channel, inter_channel, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inter_channel, 1*self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def _set_anchor(self):
        # set default anchors for each features(fpn) or just one feature(backbone)
        for idx, default_anchor in enumerate(generate_anchor_form(self.anchor, self.feature_size, self.image_size)):
            self.register_buffer('default_anchor_bbox%d'%idx, default_anchor) # [k, H, W, 4]

    def forward(self, features):
        cls_score = []
        regression_variables = []
        for idx, feature in enumerate(features):
            inter_feature = F.relu(self.inter_conv(feature))
            cls_score.append(torch.sigmoid(self.cls_conv(inter_feature))) # [B, 1*k, H, W]
            regression_variables.append(self.reg_conv(inter_feature))     # [B, 4*k, H, W]

        return {'cls_out': cls_score, 'reg_out': regression_variables}

    def get_anchor_label(self, gt, regression_variables):
        '''
        Args:
            gt (Dict) : {img, label, bbox}
                bbox (Tensor) : [B, N, 4]
            regression_variables (Tensor) : List([B, 4*k, H, W])
               
        Returns:
            anchor_label (Dict) : {anchor_bbox, anchor_pos_label, anchor_neg_label, highest_gt}
                anchor_bbox (List) : Tensor[B, k, H, W, 4] # xywh
                anchor_pos_label (List) : Tensor[B, k, H, W] label True if anchor is selected as positve sample (highest, > Threshold)
                anchor_neg_label (List) : Tensor[B, k, H, W] label True if anchor is selected as negative sample (< Threshold)
                highest_gt (List) : Tensor[P, 2] (batch, object number)  
        '''
        # initialize
        bbox = gt['bbox']
        batch_size = bbox.size()[0]
        object_num = bbox.size()[1]

        moved_anchor_bbox_list = []
        anchor_pos_label_list = []
        anchor_neg_label_list = []
        highest_gt_list = []

        kHW_list = []
        cross_IoU_list = []

        for f_idx, one_regression_variables in enumerate(regression_variables):
            feature_H = one_regression_variables.size()[2]
            feature_W = one_regression_variables.size()[3]

            kHW = self.num_anchor * feature_H * feature_W
            kHW_list.append([self.num_anchor, feature_H, feature_W])
            anchor_pos_label = bbox.new_zeros((batch_size, kHW), dtype=torch.bool)
            anchor_neg_label = bbox.new_zeros((batch_size, kHW), dtype=torch.bool)

            # anchor_bbox expand as batch size
            anchor_bbox = getattr(self, 'default_anchor_bbox%d'%f_idx).repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]

            # box regression
            if one_regression_variables != None:
                moved_anchor_bbox = box_regression(anchor_bbox, one_regression_variables, self.reg_weight) # [B, k, H, W, 4]
            else:
                moved_anchor_bbox = anchor_bbox

            if object_num != 0:
                # expand bboxes for cross calculate
                anchor_bbox_cross = moved_anchor_bbox.repeat(object_num,1,1,1,1,1).permute(1,0,2,3,4,5) # [B, k, H, W, 4] -> [B, N, k, H, W, 4]
                gt_bbox_cross = bbox.repeat(self.num_anchor, feature_H, feature_W, 1, 1, 1) # [B, N, 4] -> [k, H, W, B, N, 4]
                gt_bbox_cross = gt_bbox_cross.permute((3,4,0,1,2,5)) # [B, N, k, H, W, 4]

                # calculate IoU
                cross_IoU = IoU(anchor_bbox_cross, gt_bbox_cross).view((batch_size, object_num, kHW)) # [B, N, kHW]
                cross_IoU_list.append(cross_IoU)

                # label positive, negative anchor ()
                anchor_pos_label += (cross_IoU > self.pos_thres).any(1) # [B, kHW]
                anchor_neg_label += torch.logical_not((cross_IoU > self.neg_thres).any(1)) # [B, kHW]

                anchor_pos_label = anchor_pos_label.view((batch_size, self.num_anchor, feature_H, feature_W)) # [B, k, H, W]
                anchor_neg_label = anchor_neg_label.view((batch_size, self.num_anchor, feature_H, feature_W)) # [B, k, H, W]

            else:
                anchor_pos_label = anchor_pos_label.view((batch_size, self.num_anchor, feature_H, feature_W))
                anchor_neg_label = torch.logical_not(anchor_neg_label).view((batch_size, self.num_anchor, feature_H, feature_W))
                highest_gt = gt.new_zeros((0, 2), dtype=torch.int)

            moved_anchor_bbox_list.append(moved_anchor_bbox)
            anchor_pos_label_list.append(anchor_pos_label)
            anchor_neg_label_list.append(anchor_neg_label)

        # closest anchor
        total_cross_IoU = torch.cat(cross_IoU_list, 2) # [B, N, sum(kHW)]
        total_closest_indices = torch.argmax(total_cross_IoU, dim=2) # [B, N]
        sum_of_kHW = sum([k*h*w for k, h, w in kHW_list])
        total_one_hot_label = F.one_hot(total_closest_indices, num_classes=sum_of_kHW) # [B, N, sum(kHW)]
        
        for idx, kHW in enumerate(kHW_list):
            k, H, W = kHW
            stt, end = sum([k*h*w for k, h, w in kHW_list[:idx]]), sum([k*h*w for k, h, w in kHW_list[:idx+1]])

            # label closest anchor
            one_hot_label = total_one_hot_label[:,:,stt:end]
            closest_label = torch.sum(one_hot_label, dim=1, dtype=torch.bool) # [B, kHW]
            closest_label = closest_label.view(batch_size, k, H, W)
            anchor_pos_label_list[idx] += closest_label

            # remove closest negative anchor
            anchor_neg_label_list[idx] = torch.logical_and(torch.logical_not(closest_label), anchor_neg_label_list[idx])

            # find hightest gt bbox of each anchor
            highest_gt_per_anchor = torch.argmax(total_cross_IoU[:,:,stt:end], dim=1).view(batch_size, k, H, W) # [B, k, H, W] (int:0~N-1)
            highest_gt_object = highest_gt_per_anchor[anchor_pos_label_list[idx]] # [P] (int:0~N-1)
            highest_gt_batch  = torch.arange(0, batch_size).cuda().repeat(k, H, W, 1).permute(3,0,1,2)[anchor_pos_label_list[idx]] # [P] (int:0~B-1)
            highest_gt = torch.stack([highest_gt_batch, highest_gt_object], dim=1)
            highest_gt_list.append(highest_gt)
        

        # return dictionary
        return_dict = dict()
        return_dict['anchor_bbox'] = moved_anchor_bbox_list
        return_dict['anchor_pos_label'] = anchor_pos_label_list
        return_dict['anchor_neg_label'] = anchor_neg_label_list
        return_dict['highest_gt'] = highest_gt_list

        return return_dict

    def get_proposed_RoI(self, cls_score, regression_variables, threshold):
        '''
        R : number of proposed RoI
        Args:
            cls_score (Tensor) : List([B, 1*k, H, W])
            regression_variables (Tensor) : List([B, 4*k, H, W])
        Returns:
            RoI_bbox (Tensor) : [R, 4]
        '''
        candidate_bbox = []
        candidate_score = []
        for idx, one_cls_score in enumerate(cls_score):
            batch_size = one_cls_score.size()[0]

            # anchor_bbox expand as batch size
            #anchor_bbox = self.default_anchor_bbox.repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]
            anchor_bbox = getattr(self, 'default_anchor_bbox%d'%idx).repeat(batch_size,1,1,1,1) # [k, H, W, 4] -> [B, k, H, W, 4]

            # box regression
            if regression_variables != None:
                moved_anchor_bbox = box_regression(anchor_bbox, regression_variables[idx], self.reg_weight) # [B, k, H, W, 4]
            else:
                moved_anchor_bbox = anchor_bbox

            RoI_bool = one_cls_score > threshold
            candidate_bbox.append(moved_anchor_bbox[RoI_bool])
            candidate_score.append(one_cls_score[RoI_bool])
        RoI_bbox = nms(torch.cat(candidate_bbox), torch.cat(candidate_score), self.nms_thres)

        return RoI_bbox

    def RPN_label_select(self, cls_out, anchor_label, sample_number):
        '''
        random sampling positive, negative anchor with limited number
        P : number of positive anchor
        N : number of negative anchor
        Args:
            cls_out : List(Tensor[B, k, H, W])
            anchor_label : dict({anchor_pos_label, anchor_neg_label})
                anchor_pos_label : List(Tensor[B, k, H, W]) (bool)
                anchor_neg_label : List(Tensor[B, k, H, W]) (bool)
        Return:
            sampled_cls_out : Tensor[sample_number]
            label : Tensor[P+N] (front:ones, back:zeros)
        '''
        pos_cls_out_list = []
        neg_cls_out_list = []

        for idx, one_cls_out in enumerate(cls_out):
            pos_label, neg_label = anchor_label['anchor_pos_label'][idx], anchor_label['anchor_neg_label'][idx]

            # gathering whole positive and negative class out
            pos_cls_out_list.append(one_cls_out[pos_label])
            neg_cls_out_list.append(one_cls_out[neg_label])

        pos_cls_out, neg_cls_out = torch.cat(pos_cls_out_list), torch.cat(neg_cls_out_list)
        pos_num,     neg_num     = pos_cls_out.size()[0], neg_cls_out.size()[0]

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

        label = torch.cat([pos_cls_out.new_ones((sampled_pos_num)), pos_cls_out.new_zeros((sampled_neg_num))])

        return sampled_cls_out, label

    def RPN_cal_t_regression(self, reg_out, gt, anchor_label):
        '''
        P : number of positive anchor
        Args:
            reg_out (Tensor) : List([B, 4*k, H, W])
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
        predicted_t = []
        calculated_t = []
        for idx, one_reg_out in enumerate(reg_out):
            anchor_bbox, pos_label, highest_gt = anchor_label['anchor_bbox'][idx], anchor_label['anchor_pos_label'][idx], anchor_label['highest_gt'][idx]

            B, k4, H, W = one_reg_out.size()
            k = int(k4/4)

            # reshape reg_out for predicted_t
            predicted_t.append(one_reg_out.view(B, k, 4, H, W).permute(0,1,3,4,2)[pos_label]) # [P, 4]

            # calculate box regression parameter
            pos_anchor_bbox = anchor_bbox[pos_label] # [P, 4] (xywh)
            pos_gt_bbox = torch.stack([gt['bbox'][batch_num][gt_num] for batch_num, gt_num in highest_gt]) # [P, 4] (xywh)

            calculated_t.append(calculate_regression_parameter(pos_anchor_bbox, pos_gt_bbox, self.reg_weight))

        return torch.cat(predicted_t), torch.cat(calculated_t)

    def _get_true_anchors(self, gt, reg_out):
        true_bbox_list = []
        anchor_label = self.get_anchor_label(gt, reg_out)
        for idx, bbox in enumerate(anchor_label['anchor_bbox']):
            true_bbox_list.append(bbox[anchor_label['anchor_pos_label'][idx]])
        return torch.cat(true_bbox_list)


if __name__ == "__main__":
    xywh0 = torch.Tensor([1., 1., 1., 1.])
    xywh1 = torch.Tensor([0., 0., 3., 3.])

    print(xywh0)
    print(xywh1)
    print(IoU(xywh0, xywh1))