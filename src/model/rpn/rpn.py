import torch
import torch.nn.functional as F
import torch.nn as nn

from src.model.rpn.rpn_head import RPNHead
from src.model.anchor_func import ( generate_anchor_form, 
                                    box_regression, 
                                    anchor_preprocessing,
                                    calculate_regression_parameter,
                                    anchor_labeling_per_batch,
                                    training_anchor_selection_per_batch,
                                    training_bbox_regression_calculation,
                                    reshape_output,
                                    nms )

class RPN(nn.Module):
    '''
    F: number of feature map
    B: batch size
    A: number of entire anchors
    N: number of object
    k: number of anchors per feature
    W: width of anchor feature map
    H: height of anchor feature map
    P: number of positive anchor
    R: number of proposed RoI
    '''

    def __init__(self, FPN_mode, feature_channel, feature_size, conf_RPN):
        super().__init__()

        # init rpn head
        self.rpn_head = RPNHead(FPN_mode, feature_channel, conf_RPN)

        # init configuration values
        self.pos_thres = conf_RPN['positive_threshold']
        self.neg_thres = conf_RPN['negative_threshold']
        self.reg_weight = conf_RPN['regression_weight']
        self.nms_thres = conf_RPN['nms_threshold']
        self.k = conf_RPN['pre_nms_top_k']
        self.N = conf_RPN['proposal_N']
        self.proposal_thres = conf_RPN['proposal_threshold']

        self.feature_size = feature_size
        self.image_size = conf_RPN['input_size']

        # make default anchor set
        self.anchor = self.rpn_head.get_anchor_type()
        for idx, default_anchor in enumerate(generate_anchor_form(self.anchor, self.feature_size, self.image_size)):
            self.register_buffer('default_anchor_bbox%d'%idx, default_anchor) # [k, H, W, 4]

    def forward(self, features):
        rpn_out = self.rpn_head(features)
        #rpn_out['rpn_RoI'] = self.region_proposal_top_N(rpn_out['rpn_cls_score'], rpn_out['rpn_bbox_pred'], self.N)
        return rpn_out

    def forward_threshold(self, features):
        rpn_out = self.rpn_head(features)
        #rpn_out['rpn_RoI'] = self.region_proposal_threshold(rpn_out['rpn_cls_score'], rpn_out['rpn_bbox_pred'], self.proposal_thres)
        return rpn_out

    def get_anchor_label(self, gt_bbox, cls_score, bbox_pred):
        '''
        Args:
            gt_bbox (Tensor) : [B, N, 4]
            cls_score (List) : List([B, 1*k, H, W])
            bbox_pred (List) : List([B, 4*k, H, W])
        '''
        batch_size, _, _ = gt_bbox.size()

        # get initial anchors
        anchor_list = []
        for idx in range(len(self.anchor)):
            anchor_list.append(getattr(self, 'default_anchor_bbox%d'%idx).detach().view(-1,4).repeat(batch_size,1,1)) # [B, A, 4]

        # anchor preprocessing (top_k, bbox regression, nms etc.)
        o_anc, p_anc, p_cls, p_bbox, p_keep = anchor_preprocessing(anchor_list, cls_score, bbox_pred, self.k, self.reg_weight, self.nms_thres)

        anchor_info = dict()
        anchor_info['origin_anchors'] = o_anc
        anchor_info['anchors'] = p_anc
        anchor_info['cls_score'] = p_cls
        anchor_info['bbox_pred'] = p_bbox
        anchor_info['keep_map'] = p_keep
        anchor_info['anchor_label'], anchor_info['closest_gt'] = anchor_labeling_per_batch(anchor_info['anchors'], anchor_info['keep_map'], gt_bbox, self.pos_thres, self.neg_thres)

        return anchor_info

    def get_cls_output_target(self, cls_score, anchor_label, sample_number):
        return training_anchor_selection_per_batch(cls_score, anchor_label, sample_number)

    def get_box_output_target(self, gt_bbox, bbox_pred, anchor_label, closest_gt):
        return training_bbox_regression_calculation(gt_bbox, bbox_pred, anchor_label, closest_gt)

    def region_proposal_top_N(self, cls_score, bbox_pred, N):
        raise NotImplementedError

    def region_proposal_threshold(self, cls_score, bbox_pred, threshold):
        '''
        Args:
            cls_score (Tensor) : List([B, 1*k, H, W])
            bbox_pred (Tensor) : List([B, 4*k, H, W])
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
            if bbox_pred != None:
                moved_anchor_bbox = box_regression(anchor_bbox, bbox_pred[idx], self.reg_weight) # [B, k, H, W, 4]
            else:
                moved_anchor_bbox = anchor_bbox

            RoI_bool = one_cls_score > threshold
            candidate_bbox.append(moved_anchor_bbox[RoI_bool])
            candidate_score.append(one_cls_score[RoI_bool])
        RoI_bbox = nms(torch.cat(candidate_bbox), torch.cat(candidate_score), self.nms_thres)

        return RoI_bbox

    def get_anchor_label_deprecated(self, gt, regression_variables):
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

    def _get_positive_anchors(self, gt, bbox_pred):
        raise NotImplementedError
        return positive_anchors
