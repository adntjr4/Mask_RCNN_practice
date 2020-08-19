
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, in_channel, inter_channel):
        super().__init__()

        self._set_anchor()
        self.num_anchor = len(self.anchor)

        self.inter_conv = nn.Conv2d(in_channel, inter_channel, kernel_size=3)
        self.cls_conv = nn.Conv2d(inter_channel, 2*self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def _set_anchor(self):
        self.anchor = []
        for size in [128, 256, 512]:
            for ratio in [0.5, 1, 2]:
                self.anchor.append({size, ratio})

    def forward(self, features):
        inter_feature = self.inter_conv(features)
        cls_score = self.cls_conv(inter_feature)
        reg_coor = self.reg_conv(inter_feature)

        return cls_score, reg_coor
