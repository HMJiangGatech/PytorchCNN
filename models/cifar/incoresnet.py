from __future__ import absolute_import

'''SNNet

WResnet is 4x wider
'''
import torch.nn as nn
import math
from ..IncoConv import IncoConv2d


__all__ = ['inco_resnet','inco_resnet110','inco_resnet56','inco_resnet44','inco_resnet32','inco_resnet20',
            'inco_wresnet20','inco_wresnet32','inco_wresnet44','inco_wresnet110']

def incoconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return IncoConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class IncoBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IncoBasicBlock, self).__init__()
        self.conv1 = incoconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = incoconv3x3(planes, planes)
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


class IncoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IncoBottleneck, self).__init__()
        self.conv1 = IncoConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = IncoConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = IncoConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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


class IncoResNet(nn.Module):

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10):
        super(IncoResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = IncoBottleneck if depth >=44 else IncoBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = IncoConv2d(3, self.inplanes, kernel_size=3, padding=1,
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
                IncoConv2d(self.inplanes, planes * block.expansion,
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

    def orth_reg(self):
        reg = 0
        for m in self.modules():
            if isinstance(m, Orth_Conv2d):
                reg += m.orth_reg()
        return reg

    def project(self):
        for m in self.modules():
            if isinstance(m, IncoConv2d):
                m.project()

    def showOrthInfo(self):
        for m in self.modules():
            if isinstance(m, IncoConv2d):
                m.showOrthInfo()


def inco_resnet(**kwargs):
    """
    Constructs a IncoResNet model.
    """
    return IncoResNet(**kwargs)

def inco_resnet110(**kwargs):
    """Constructs a IncoResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(110, **kwargs)
    return model

def inco_resnet56(**kwargs):
    """Constructs a IncoResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(56, **kwargs)
    return model

def inco_resnet44(**kwargs):
    """Constructs a IncoResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(44, **kwargs)
    return model

def inco_resnet32(**kwargs):
    """Constructs a IncoResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(32, **kwargs)
    return model

def inco_resnet20(**kwargs):
    """Constructs a IncoResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(20, **kwargs)
    return model

def inco_wresnet20(**kwargs):
    """Constructs a Wide IncoResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def inco_wresnet32(**kwargs):
    """Constructs a Wide IncoResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(32, nfilter = [64,128,256], **kwargs)
    return model

def inco_wresnet44(**kwargs):
    """Constructs a Wide IncoResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def inco_wresnet110(**kwargs):
    """Constructs a Wide IncoResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IncoResNet(110, nfilter = [64,128,256], **kwargs)
    return model
