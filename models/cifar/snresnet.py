from __future__ import absolute_import

'''SphereNet

WResnet is 4x wider
'''
import torch.nn as nn
import math
from ..SNConv import SNConv2d


__all__ = ['sn_resnet','sn_resnet110','sn_resnet56','sn_resnet44','sn_resnet32','sn_resnet20',
            'sn_wresnet20','sn_wresnet32','sn_wresnet44','sn_wresnet110']

def snconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SNConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNBasicBlock, self).__init__()
        self.conv1 = snconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = snconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNBottleneck, self).__init__()
        self.conv1 = SNConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SNConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SNConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SphereResNet(nn.Module):

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10):
        super(SphereResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SNBottleneck if depth >=44 else SNBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = SNConv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nfilter[0], n)
        self.layer2 = self._make_layer(block, nfilter[1], n, stride=2)
        self.layer3 = self._make_layer(block, nfilter[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nfilter[2] * block.expansion, num_classes)

        self.features = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,    # 32x32
                self.layer1,  # 32x32
                self.layer2,  # 16x16
                self.layer3,  # 8x8
                self.avgpool
        )

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SNConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def project(self):
        for m in self.modules():
            if isinstance(m, SNConv2d):
                m.project()

    def showOrthInfo(self):
        for m in self.modules():
            if isinstance(m, SNConv2d):
                m.showOrthInfo()


def sn_resnet(**kwargs):
    """
    Constructs a SphereResNet model.
    """
    return SphereResNet(**kwargs)

def sn_resnet110(**kwargs):
    """Constructs a SphereResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(110, **kwargs)
    return model

def sn_resnet56(**kwargs):
    """Constructs a SphereResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(56, **kwargs)
    return model

def sn_resnet44(**kwargs):
    """Constructs a SphereResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(44, **kwargs)
    return model

def sn_resnet32(**kwargs):
    """Constructs a SphereResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(32, **kwargs)
    return model

def sn_resnet20(**kwargs):
    """Constructs a SphereResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(20, **kwargs)
    return model

def sn_wresnet20(**kwargs):
    """Constructs a Wide SphereResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def sn_wresnet32(**kwargs):
    """Constructs a Wide SphereResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(32, nfilter = [64,128,256], **kwargs)
    return model

def sn_wresnet44(**kwargs):
    """Constructs a Wide SphereResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def sn_wresnet110(**kwargs):
    """Constructs a Wide SphereResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(110, nfilter = [64,128,256], **kwargs)
    return model
