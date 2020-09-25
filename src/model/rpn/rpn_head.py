
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.util.util import IoU
from src.model.anchor_func import generate_anchor_form, box_regression, calculate_regression_parameter


class RPNHead(nn.Module):
    def __init__(self, FPN_mode, feature_channel, conf_RPN):
        super().__init__()

        self.FPN_mode = FPN_mode

        self.anchor_size = conf_RPN['anchor_size']
        self.anchor_ratio = conf_RPN['anchor_ratio']
        self.num_anchor = self.get_anchor_number()
        
        # build head layers
        inter_channel = conf_RPN['intermediate_channel_number']
        self.inter_conv = nn.Conv2d(feature_channel, inter_channel, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inter_channel, 1*self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def get_anchor_type(self):
        anchor = []
        if self.FPN_mode:
            for size in self.anchor_size:
                anchor.append([[size, ratio] for ratio in self.anchor_ratio])
        else:
            for size in self.anchor_size:
                for ratio in self.anchor_ratio:
                    anchor.append([size, ratio])
            anchor = [anchor]
        return anchor

    def get_anchor_number(self):
        if self.FPN_mode:
            return len(self.anchor_ratio)
        else:
            return len(self.anchor_size) * len(self.anchor_ratio)

    def forward(self, features):
        cls_score = []
        bbox_pred = []
        for idx, feature in enumerate(features):
            inter_feature = F.relu(self.inter_conv(feature))
            cls_score.append(torch.sigmoid(self.cls_conv(inter_feature))) # [B, 1*k, H, W]
            bbox_pred.append(self.reg_conv(inter_feature))     # [B, 4*k, H, W]

        return {'rpn_cls_score': cls_score, 'rpn_bbox_pred': bbox_pred}
