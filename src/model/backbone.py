import torch
import torch.nn as nn
import torchvision.models as models

class BackBone(nn.Module):
    def __init__(self, backbone:str='R50'):
        super().__init__()
        
        assert backbone in ['R50', 'R101']
        self.backbone = backbone
        
        self._build_model()

    def _build_model(self):
        if self.backbone == 'R50': # 2048x7x7
            self.model = models.resnet50(pretrained=True)
        elif self.backbone == 'R101': # 2048x7x7
            self.model = models.resnet101(pretrained=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def get_channel(self):
        return 2048, 7, 7


if __name__ == '__main__':
    import torch
    bb = BackBone()
    x = torch.randn(1, 3, 1024, 1024)
    y = bb(x)
    print(y.size())
