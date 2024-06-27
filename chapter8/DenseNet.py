import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, 
        num_channels=64, 
        growth_rate=32, 
        arch=(4, 4, 4, 4),
        num_classes=10
    ):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(self.stem())
        for i, num_convs in enumerate(arch):
            self.net.append(DenseBlk(num_convs, growth_rate))

            num_channels += num_convs * growth_rate

            if i != len(arch)-1:
                num_channels //= 2
                self.net.append(self.transition_block(num_channels))
        self.net.append(self.head(num_classes))

    def stem(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def transition_block(self, num_channels):
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def head(self, num_classes):
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, X):
        return self.net(X)


class DenseBlk(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlk, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def conv_block(self, num_channels):
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(num_channels, 3, padding=1)
        )

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
