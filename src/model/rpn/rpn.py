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
                                    training_anchor_selection_per_batch,
                                    training_bbox_regression_calculation,
                                    reshape_output,
                                    nms_per_batch )
from src.util.util import transform_xywh

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
        self.pos_thres      = conf_RPN['positive_threshold']
        self.neg_thres      = conf_RPN['negative_threshold']
        self.reg_weight     = conf_RPN['regression_weight']
        self.nms_thres      = conf_RPN['nms_threshold']
        self.sample_number  = conf_RPN['RPN_sample_number']
        self.pre_k          = conf_RPN['pre_nms_top_k']
        self.post_k         = conf_RPN['post_nms_top_k']
        self.proposal_thres = conf_RPN['proposal_threshold']
        

        self.feature_size = feature_size
        self.image_size = conf_RPN['input_size']

        # make default anchor set
        self.anchor = self.rpn_head.get_anchor_type()
        #for idx, default_anchor in enumerate(generate_anchor_form(self.anchor, self.feature_size, self.image_size)):
        #    self.register_buffer('default_anchor_bbox%d'%idx, default_anchor) # [k, H, W, 4]

    def forward(self, features):
        rpn_out = self.rpn_head(features)
        #rpn_out['rpn_RoI'] = self.region_proposal_top_N(rpn_out['rpn_cls_score'], rpn_out['rpn_bbox_pred'], self.N)
        return rpn_out

    def forward_threshold(self, features):
        rpn_out = self.rpn_head(features)
        #rpn_out['rpn_RoI'] = self.region_proposal_threshold(rpn_out['rpn_cls_score'], rpn_out['rpn_bbox_pred'], self.proposal_thres)
        return rpn_out

    def anchor_preparing(self, image_size, cls_score, bbox_pred):
        '''
        Args:
            image_size (Tensor) : [B, 2]
            cls_score (List) : List([B, 1*k, H, W])
            bbox_pred (List) : List([B, 4*k, H, W])
        Returns:
            origin_anchors (Tensor) : [B, A, 4]
            post_anchors (Tensor) : [B, A, 4]
            post_cls_score (Tensor) : [B, A, 1]
            post_bbox_pred (Tensor) : [B, A, 4]
            keep_map (Tensor) : [B, A]
        '''
        batch_size, _, _, _ = cls_score[0].size()

        # get initial anchors
        anchor_list = generate_anchor_form(self.anchor, self.feature_size, self.image_size)
        for idx, anchor in enumerate(anchor_list):
            anchor_list[idx] = anchor.view(-1,4).repeat(batch_size,1,1) # [B, A, 4]
            #anchor_list.append(getattr(self, 'default_anchor_bbox%d'%idx).detach().view(-1,4).repeat(batch_size,1,1)) # [B, A, 4]

        # anchor preprocessing (top_k, bbox regression, nms etc.)
        return anchor_preprocessing(anchor_list, image_size, cls_score, bbox_pred, self.pre_k, self.post_k, self.reg_weight, self.nms_thres)

    def get_anchor_label(self, gt_bbox, image_size, cls_score, bbox_pred):
        '''
        Args:
            gt_bbox (Tensor) : [B, N, 4]
            image_size (Tensor) : [B, 2]
            cls_score (List) : List([B, 1*k, H, W])
            bbox_pred (List) : List([B, 4*k, H, W])
        Returns:
            anchor_info (Dict)
        '''
        o_anc, p_anc, p_cls, p_bbox = self.anchor_preparing(image_size, cls_score, bbox_pred)

        anchor_info = dict()
        anchor_info['origin_anchors'] = o_anc
        anchor_info['anchors'] = p_anc
        anchor_info['cls_score'] = p_cls
        anchor_info['bbox_pred'] = p_bbox

        anchor_info['anchor_label'], anchor_info['closest_gt'] = anchor_labeling_per_batch(anchor_info['anchors'], gt_bbox, self.pos_thres, self.neg_thres)

        return anchor_info

    def get_cls_output_target(self, cls_score, anchor_label):
        return training_anchor_selection_per_batch(cls_score, anchor_label, self.sample_number)

    def get_box_output_target(self, gt_bbox, origin_anchors, bbox_pred, anchor_label, closest_gt):
        return training_bbox_regression_calculation(gt_bbox, origin_anchors, bbox_pred, anchor_label, closest_gt, self.reg_weight)

    def region_proposal_top_N(self, cls_score, bbox_pred, N):
        '''
        propose RoIs using top-N
        Args:
            cls_score (Tensor) : List([B, 1*k, H, W])
            bbox_pred (Tensor) : List([B, 4*k, H, W])
            N (int)
        Returns:
            RoI_bbox (Tensor) : [N, 4]
        '''
        raise NotImplementedError

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
            nms_thres (float) : nms threshold for detection proposal
        Returns:
            bboxes (Tensor) : [N, 4]
            scores (Tensor) : [N, 1]
            img_id_map (Tensor) : [N]
        '''
        _, post_anchors, post_cls_score, _ = self.anchor_preparing(image_size, cls_score, bbox_pred)
        over_score_map = post_cls_score.squeeze(-1) > threshold

        # proposal nms
        nms_keep = nms_per_batch(post_anchors, post_cls_score, nms_thres)

        return_proposal_keep = torch.logical_and(over_score_map, nms_keep)

        bboxes = []
        for idx in range(len(img_id)):
            transfromed_xywh = transform_xywh((post_anchors[idx][return_proposal_keep[idx]]).cpu(), np.array(inv_trans[idx].cpu()))
            bboxes.append(post_anchors.new(transfromed_xywh))
        bboxes = torch.cat(bboxes)

        scores = post_cls_score[return_proposal_keep]
        
        img_id_map = torch.cat([one_id.repeat(return_proposal_keep[idx].sum()) for idx, one_id in enumerate(img_id)])

        return bboxes, scores, img_id_map
        
    def _get_positive_anchors(self, anchors, label):
        return anchors[label > 0.1]
