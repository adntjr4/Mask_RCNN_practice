
import torch
import torch.nn as nn

from src.model.backbone.backbone import BackBone
from src.model.backbone.fpn import FPN
from src.model.rpn.rpn import RPN
from src.model.rpn.rpn_head import RPNHead


class BaseModel(nn.Module):
    def __init__(self, conf_model):
        super().__init__()

        self.conf_backbone = conf_model['backbone']
        self.conf_RPN = conf_model['RPN']
        self.RPN_sample_number = conf_model['RPN']['RPN_sample_number']

        self._build_backbone()
        self._build_RPN()
        self._set_criterion()

    def _build_backbone(self):
        if 'FPN' in self.conf_backbone:
            self.backbone = FPN(self.conf_backbone)
        else:
            self.backbone = BackBone(self.conf_backbone)

    def _build_RPN(self):
        input_size = self.conf_backbone['input_size']
        self.RPN = RPN( 'FPN' in self.conf_backbone,
                        self.backbone.get_feature_channel(), 
                        self.backbone.get_feature_size(input_size),
                        self.conf_RPN)

    def _set_criterion(self):
        # RPN losses (cls_loss, reg_loss)
        self.rpn_cls_criterion = nn.BCELoss(reduction='sum') # log loss
        self.rpn_box_criterion = nn.SmoothL1Loss(reduction='sum') # robust loss

        # and something more...

    def forward(self, data, mode):
        '''
        Entire model forward function.
        Args:
            data (dict) : input of model
            mode (str) : 'train' or 'eval'
        Returns (train):
            losses
        Returns (eval):
            bbox (Tensor) : [sum(N), 4]
            img_id : [sum(N)]
        '''
        model_out = dict()
        feature_map = self.backbone(data['img'])
        RPN_out = self.RPN(feature_map)
        model_out.update(RPN_out)

        # training mode (return losses)
        if mode == 'train':
            loss_switch = {'rpn_cls':True, 'rpn_box':True}

            # get losses
            losses = {}

            # [RPN] anchor labeling
            anchor_info = self.RPN.get_anchor_label(data['bbox'], data['img_size'], model_out['rpn_cls_score'], model_out['rpn_bbox_pred'])

            #### FOR DEBUGGING : POSITIVE LABEL IMAGE OUT
            # from src.util.debugger import debug_draw_bbox3_cv_img
            # img0_gt = data['bbox'][0]
            # img0_ori = self.RPN._get_positive_anchors(anchor_info['origin_anchors'][0], anchor_info['anchor_label'][0])
            # img0_anc = self.RPN._get_positive_anchors(anchor_info['anchors'][0], anchor_info['anchor_label'][0])

            # img1_gt = data['bbox'][1]
            # img1_ori = self.RPN._get_positive_anchors(anchor_info['origin_anchors'][1], anchor_info['anchor_label'][1])
            # img1_anc = self.RPN._get_positive_anchors(anchor_info['anchors'][1], anchor_info['anchor_label'][1])

            # debug_draw_bbox3_cv_img(data['img'][0], img0_gt, img0_ori, img0_anc, 'img0')
            # debug_draw_bbox3_cv_img(data['img'][1], img1_gt, img1_ori, img1_anc, 'img1')
            ##### FOR DEBUGGING ##########################

            RPN_normalizer = data['bbox'].size()[0] * self.RPN_sample_number

            # [RPN] class loss
            if loss_switch['rpn_cls']:
                selected_cls_out, label = self.RPN.get_cls_output_target(anchor_info['cls_score'], anchor_info['anchor_label'])
                losses['rpn_cls_loss'] = self.rpn_cls_criterion(selected_cls_out, label) / RPN_normalizer

            # [RPN] bbox regression loss
            if loss_switch['rpn_box']:
                if anchor_info['closest_gt'].size()[0] != 0:
                    predicted_t, calculated_t = self.RPN.get_box_output_target(data['bbox'], anchor_info['origin_anchors'], anchor_info['bbox_pred'], anchor_info['anchor_label'], anchor_info['closest_gt'])
                    losses['rpn_box_loss'] = self.rpn_box_criterion(predicted_t, calculated_t) / RPN_normalizer
                else:
                    losses['rpn_box_loss'] = data['bbox'].new_zeros(())

            return losses

        # evaluation mode (return bboxes, scores ...)
        else:
            bboxes, scores, img_id_map = self.RPN.region_proposal_threshold(
                data['img_size'], 
                model_out['rpn_cls_score'], 
                model_out['rpn_bbox_pred'], 
                data['img_id'], 
                data['inv_trans'], 
                self.conf_RPN['proposal_threshold'],
                self.conf_RPN['proposla_nms_threshold'])
            '''
            origin_bboxes, bboxes, scores, img_id_map = self.RPN.region_proposal_top_N(
                data['img_size'], 
                model_out['rpn_cls_score'], 
                model_out['rpn_bbox_pred'], 
                data['img_id'], 
                data['inv_trans'], 
                100)
            '''

            return bboxes, scores, img_id_map

    def get_parameters(self):
        return  list(self.backbone.get_parameters()) + \
                list(self.RPN.parameters())

