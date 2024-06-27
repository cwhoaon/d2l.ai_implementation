import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super(ResNet, self).__init__()
        
        self.net = self.begin_blk()
        for i, params in enumerate(arch):
            self.net.append(self.res_blk(*params, i==0))
        self.net.append(self.last_blk(num_classes))

    def begin_blk(self):
        return nn.Sequential(
            nn.LazyConv2d(64, 7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
    
    def res_blk(self, num_residuals, num_channels, first_blk=False):
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_blk:
                blk.append(Residual_blk(num_channels, False, 2))
            else:
                blk.append(Residual_blk(num_channels))
        return nn.Sequential(*blk)
    
    def last_blk(self, num_classes):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.net(x)


class Residual_blk(nn.Module):
    def __init__(self, num_channels, maintain_channels=True, strides=1):
        super(Residual_blk, self).__init__()
        self.conv1 = nn.LazyConv2d(num_channels, 3, stride=strides, padding=1)
        self.conv2 = nn.LazyConv2d(num_channels, 3, padding=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        if not maintain_channels:
            self.conv3 = nn.LazyConv2d(num_channels, 1, stride=strides)
        else:
            self.conv3 = None
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(X)
