import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FPN(nn.Module):
    def __init__(self, conf_backbone):
        super().__init__()
        
        self.backbone = conf_backbone['backbone_type']
        assert self.backbone in ['R50', 'R101']
        self.ch_number = conf_backbone['feature_channel_number']

        self.features_name = ['p2', 'p3', 'p4', 'p5', 'p6']

        self._build_model()

    def _build_model(self):
        if self.backbone == 'R50':
            self.model = models.resnet50(pretrained=True)
        elif self.backbone == 'R101':
            self.model = models.resnet101(pretrained=True)

        #for param in self.model.parameters():
        #    param.requires_grad = False

        in_channel = [2048, 1024, 512, 256]
        self.lateral_conv = nn.ModuleList([nn.Conv2d(in_channel[i], self.ch_number, kernel_size=1, stride=1, padding=0) for i in range(4)])
        self.output_conv = nn.ModuleList([nn.Conv2d(self.ch_number, self.ch_number, kernel_size=3, stride=1, padding=1) for i in range(4)])

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        c2 = self.model.layer1(x)
        c3 = self.model.layer2(c2)
        c4 = self.model.layer3(c3)
        c5 = self.model.layer4(c4)

        features = [c5, c4, c3, c2]

        output_features = []
        prev_feature = None
        for idx, feature in enumerate(features):
            lateral_feature = self.lateral_conv[idx](feature)
            if prev_feature is not None:
                upsample_feature = F.interpolate(prev_feature, scale_factor=2, mode="nearest")
                prev_feature = lateral_feature + upsample_feature
            else:
                prev_feature = lateral_feature
            output_features.append(self.output_conv[idx](prev_feature))
        # output_features : [p5, p4, p3, p2]

        # p6 feature
        p6 = F.max_pool2d(output_features[0], kernel_size=2)
        output_features.insert(0, p6)

        return output_features # [p6, p5, p4, p3, p2]

    def get_feature_channel(self):
        return self.ch_number

    def get_feature_size(self, input_size):
        return [(int(input_size[0]/(2**i)), int(input_size[1]/2**i)) for i in range(len(self.features_name)+1, 1, -1)]


if __name__ == '__main__':
    cfg = {'FPN': True, 'feature_channel_number':256, 'backbone_type':'R50'}
    bb = FPN(cfg)
    x = torch.randn(1, 3, 1024, 1024)
    y = bb(x)
    for i in y:
        print(i.size())
