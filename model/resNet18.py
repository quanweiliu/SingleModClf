import torch
from torch import nn
from torchvision import models
from functools import partial
import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class resNet18(nn.Module):
    def __init__(self, bands, classes=2, is_pretrained="ResNet18_Weights.DEFAULT"):
        
        super(resNet18, self).__init__()
        self.resnet = models.resnet18(weights=is_pretrained)
        self.resnet.fc = nn.Linear(512, classes)
        # print(self.resnet)

    def _initalize_weights(self):
        init_set = {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}
        for module in self.modules():
            if type(module) in init_set:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bands = 3
    x = torch.randn(4, bands, 128, 128, device=device)

    model = resNet18(bands, classes=2).to(device)
    output = model(x)
    print("output", output.shape)

