

import torch
import torch.nn as nn


class RoIHead(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, proposals):
        raise NotImplementedError
