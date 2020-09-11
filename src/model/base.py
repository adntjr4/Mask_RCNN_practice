
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
        self.rpn_cls_criterion = nn.BCELoss() # log loss
        self.rpn_box_criterion = nn.SmoothL1Loss() # robust loss

        # and something more...

    def forward(self, data, mode):
        model_out = dict()
        feature_map = self.backbone(data['img'])
        RPN_out = self.RPN(feature_map)
        model_out.update(RPN_out)

        if mode == 'train':
            loss_switch = {'rpn_cls':True, 'rpn_box':True}

            # get losses
            losses = {}

            # [RPN] anchor labeling
            anchor_info = self.RPN.get_anchor_label(data['bbox'], model_out['rpn_cls_score'], model_out['rpn_bbox_pred'])

            # [RPN] class loss
            if loss_switch['rpn_cls']:
                selected_cls_out, label = self.RPN.get_cls_output_target(anchor_info['cls_score'], anchor_info['anchor_label'])
                losses['rpn_cls_loss'] = self.rpn_cls_criterion(selected_cls_out, label)

            # [RPN] bbox regression loss
            if loss_switch['rpn_box']:
                if anchor_info['closest_gt'].size()[0] != 0:
                    predicted_t, calculated_t = self.RPN.get_box_output_target(data['bbox'], anchor_info['origin_anchors'], anchor_info['bbox_pred'], anchor_info['anchor_label'], anchor_info['closest_gt'])
                    losses['rpn_box_loss'] = self.rpn_box_criterion(predicted_t, calculated_t)
                else:
                    losses['rpn_box_loss'] = torch.tensor(0.).cuda()

            return losses
        else:
            return model_out

    def get_parameters(self):
        return  list(self.backbone.parameters()) + \
                list(self.RPN.parameters())

