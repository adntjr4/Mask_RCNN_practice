import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.model.rpn.rpn_head import RPNHead
from src.model.anchor_func import ( generate_anchor_form, 
                                    box_regression, 
                                    anchor_preprocessing,
                                    anchor_labeling_per_batch,
                                    anchor_labeling_no_gt,
                                    invaild_bbox_cliping_per_batch,
                                    calculate_regression_parameter,
                                    reshape_output,
                                    nms_per_batch,
                                    sort_per_batch,
                                    top_k_from_indices )


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
        self.pos_thres         = conf_RPN['positive_threshold']
        self.neg_thres         = conf_RPN['negative_threshold']
        self.reg_weight        = conf_RPN['regression_weight']
        self.nms_thres         = conf_RPN['nms_threshold']
        self.sampling_number   = conf_RPN['sampling_number']
        self.positive_fraction = conf_RPN['positive_fraction']
        self.pre_k             = conf_RPN['pre_nms_top_k']
        self.post_k            = conf_RPN['post_nms_top_k']

        self.feature_size = feature_size
        self.image_size = conf_RPN['input_size']

        # make default anchor set
        self.anchor_type = self.rpn_head.get_anchor_type()

        self._set_criterion()

    def _set_criterion(self):
        self.rpn_objectness_criterion = nn.BCELoss(reduction='mean') # log loss
        self.rpn_regression_criterion = nn.SmoothL1Loss(reduction='sum') # robust loss

    def forward(self, features, data, mode):
        rpn_out = self.rpn_head(features)

        origin_anchors, objectnesses, bbox_deltas = self.anchor_preparing(data['img_size'], rpn_out['rpn_objectness'], rpn_out['rpn_bbox_delta'])

        losses = dict()

        if mode == 'train':
            # anchor labeling
            anchor_label, closest_gt = anchor_labeling_per_batch(origin_anchors, data['bbox'], self.pos_thres, self.neg_thres)

            # objectness loss
            selected_cls_out, label = self.get_cls_output_target(objectnesses, anchor_label)
            losses['rpn_obj'] = self.rpn_objectness_criterion(selected_cls_out, label)

            # bbox regression loss
            if closest_gt.size()[0] != 0:
                predicted_t, calculated_t = self.get_box_output_target(data['bbox'], origin_anchors, bbox_deltas, anchor_label, closest_gt)
                losses['rpn_reg'] = self.rpn_regression_criterion(predicted_t, calculated_t)
            else:
                losses['rpn_reg'] = data['bbox'].new_zeros(())

        # bbox regression
        proposals = box_regression(origin_anchors, bbox_deltas, self.reg_weight)

        # invaild bbox clipping
        invaild_bbox_cliping_per_batch(proposals, data['img_size'])

        return proposals, losses

    def anchor_preparing(self, image_size, cls_score, bbox_pred):
        '''
        Args:
            image_size (Tensor) : [B, 2]
            cls_score (List) : List([B, 1*k, H, W])
            bbox_pred (List) : List([B, 4*k, H, W])
        Returns:
            origin_anchors (Tensor) : [B, A, 4]
            objectnesses (Tensor)   : [B, A, 1]
            bbox_deltas (Tensor)    : [B, A, 4]
        '''
        batch_size, _, _, _ = cls_score[0].size()

        # get initial anchors
        anchor_list = generate_anchor_form(self.anchor_type, self.feature_size, self.image_size)
        for idx, anchor in enumerate(anchor_list):
            anchor_list[idx] = anchor.view(-1,4).repeat(batch_size,1,1) # [B, A, 4]

        # anchor preprocessing (top_k, bbox regression, nms etc.)
        origin_anchors, objectnesses, bbox_deltas = anchor_preprocessing(anchor_list, image_size, cls_score, bbox_pred, self.pre_k, self.post_k, self.nms_thres)

        return origin_anchors, objectnesses, bbox_deltas

    def get_cls_output_target(self, cls_score, anchor_label):
        '''
        random anchor sampling for training
        Args:
            cls_score (Tensor) : [B, A, 1]
            anchor_label (Tensor) : [B, A] (1, 0, -1)
            sampling_number (int)
        returns:
            training_cls_score (Tensor) : [B, sampling_number]
            training_cls_gt : [B, sampling_number]
        '''
        batch_size = cls_score.size()[0]

        training_cls_score_list = []
        training_cls_gt_list    = []

        # for each batch
        for b_idx in range(batch_size):
            # gather objectness
            pos_cls_out, neg_cls_out = cls_score[b_idx][anchor_label[b_idx] > 0], cls_score[b_idx][anchor_label[b_idx] < 0] # [P], [N]
            pos_num,     neg_num     = pos_cls_out.size()[0],                     neg_cls_out.size()[0]

            # random sampling
            pivot = int(self.sampling_number * self.positive_fraction)
            sampled_pos_num = min(pivot, pos_num)
            sampled_neg_num = self.sampling_number - sampled_pos_num
            sampled_pos_cls_out = pos_cls_out[torch.randperm(pos_num)[:sampled_pos_num]]
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sampled_neg_num]]

            # training samples concatnate
            one_cls_score = torch.cat([sampled_pos_cls_out, sampled_neg_cls_out]).squeeze(-1)
            one_cls_gt = torch.cat([pos_cls_out.new_ones((sampled_pos_num)), pos_cls_out.new_zeros((sampled_neg_num))])

            training_cls_score_list.append(one_cls_score)
            training_cls_gt_list.append(one_cls_gt)

        return torch.stack(training_cls_score_list), torch.stack(training_cls_gt_list)

    def get_box_output_target(self, gt_bbox, origin_anchors, bbox_pred, anchor_label, closest_gt):
        '''
        Args:
            gt_bbox (Tensor) : [B, N, 4]
            origin_anchors (Tensor) : [B, A, 4]
            bbox_pred (Tensor) : [B, A, 4]
            anchor_label (Tensor) : [B, A] (1, 0, -1)
            closest_gt (Tensor) : [P, 2] (0 ~ B-1), (0 ~ N-1)
        Returns:
            predicted_t  : Tensor[P, 4]
            calculated_t : Tensor[P, 4]
        '''
        # calculate target regression parameter
        positive_anchors = origin_anchors[anchor_label>0] # [P, 4]
        positive_gt = torch.stack([gt_bbox[batch_num][gt_num] for batch_num, gt_num in closest_gt]) # [P, 4]
        calculated_t = calculate_regression_parameter(positive_anchors, positive_gt, self.reg_weight) # [P, 4]

        # reshape output regression prediction
        predicted_t = bbox_pred[anchor_label>0] # [P, 4]

        return predicted_t, calculated_t

    def region_proposal_top_N(self, image_size, cls_score, bbox_pred, img_id, inv_trans, N):
        '''
        propose RoIs using score thresholding
        Args:
            image_size (Tensor) : [B, 2]
            cls_score (Tensor) : List([B, 1*k, H, W])
            bbox_pred (Tensor) : List([B, 4*k, H, W])
            img_id (Tensor) : [B]
            inv_trans (Tensor) : [B, 2, 3]
            N (int)
        Returns:
            bboxes (Tensor) : [N, 4]
            scores (Tensor) : [N, 1]
            img_id_map (Tensor) : [N]
        '''
        # get anchors from cls score and bbox variables
        origin_anchors, post_cls_score, post_bbox_pred = self.anchor_preparing(image_size, cls_score, bbox_pred)

        # bbox regression
        post_anchors = box_regression(origin_anchors, post_bbox_pred, self.reg_weight)

        # select top N anchors
        indices = sort_per_batch(post_cls_score)
        bboxes = top_k_from_indices(post_anchors, indices, N)
        scores = top_k_from_indices(post_cls_score, indices, N)
        img_id_map = torch.cat([one_id.repeat(N) for one_id in img_id])

        # transform into orinal image space
        bboxes_list = []
        for idx in range(len(img_id)):
            transfromed_xywh = transform_xywh(bboxes[idx].cpu(), np.array(inv_trans[idx].cpu()))
            bboxes_list.append(post_anchors.new(transfromed_xywh))
        bboxes = torch.cat(bboxes_list)
        scores = torch.cat([t[0] for t in scores.split(1)])

        return bboxes, scores, img_id_map

    def region_proposal_threshold(self, image_size, cls_score, bbox_pred, img_id, inv_trans, threshold, nms_thres):
        '''
        propose RoIs using score thresholding
        Args:
            image_size (Tensor) : [B, 2]
            cls_score (Tensor) : List([B, 1*k, H, W])
            bbox_pred (Tensor) : List([B, 4*k, H, W])
            img_id (Tensor) : [B]
            inv_trans (Tensor) : [B, 2, 3]
            threshold (float)
            nms_thres (float)
        Returns:
            bboxes (Tensor) : [N, 4]
            scores (Tensor) : [N, 1]
            img_id_map (Tensor) : [N]
        '''
        # get anchors from cls score and bbox variables
        origin_anchors, post_cls_score, post_bbox_pred = self.anchor_preparing(image_size, cls_score, bbox_pred)

        # select anchors which has over threshold
        over_score_map = post_cls_score.squeeze(-1) > threshold

        # bbox regression
        post_anchors = box_regression(origin_anchors, post_bbox_pred, self.reg_weight)
        
        # proposal nms
        nms_keep = nms_per_batch(post_anchors, post_cls_score, nms_thres)
        return_proposal_keep = torch.logical_and(over_score_map, nms_keep)

        # transform into orinal image space
        bboxes = []
        for idx in range(len(img_id)):
            transfromed_xywh = transform_xywh((post_anchors[idx][return_proposal_keep[idx]]).cpu(), np.array(inv_trans[idx].cpu()))
            bboxes.append(post_anchors.new(transfromed_xywh))
        bboxes = torch.cat(bboxes)

        # corresponding score
        scores = post_cls_score[return_proposal_keep]
        
        # corresponding image id
        img_id_map = torch.cat([one_id.repeat(return_proposal_keep[idx].sum()) for idx, one_id in enumerate(img_id)])

        return bboxes, scores, img_id_map
        
    def _get_positive_anchors(self, anchors, label):
        return anchors[label > 0.1]
