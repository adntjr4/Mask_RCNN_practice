import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BackBone(nn.Module):
    def __init__(self, conf_backbone):
        super().__init__()
        
        self.backbone = conf_backbone['backbone_type']
        assert self.backbone in ['R50', 'R101']

        if self.backbone == 'R50':
            self.model = models.resnet50(pretrained=True)
        elif self.backbone == 'R101':
            self.model = models.resnet101(pretrained=True)
        
        # requires_grad off
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return [x]

    def get_parameters(self):
        return  list(self.model.layer2.parameters()) + \
                list(self.model.layer3.parameters()) + \
                list(self.model.layer4.parameters())


    def get_feature_channel(self):
        return 2048

    def get_feature_size(self, input_size):
        return [(int(input_size[0]/32), int(input_size[1]/32))]


if __name__ == '__main__':
    import torch
    bb = BackBone({'backbone_type':'R50'})
    x = torch.randn(1, 3, 1024, 1024)
    y = bb(x)
    for i in y:
        print(i.size())
