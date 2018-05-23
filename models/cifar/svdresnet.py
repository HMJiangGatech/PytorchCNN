from __future__ import absolute_import

'''SVDNet

WResnet is 4x wider
'''
import torch.nn as nn
import math
from ..SVDConv import SVDConv2d


__all__ = ['svd_resnet','svd_resnet110','svd_resnet56','svd_resnet44','svd_resnet32','svd_resnet20',
            'svd_wresnet20','svd_wresnet32','svd_wresnet44','svd_wresnet110']

def svdconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SVDConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SVDBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SVDBasicBlock, self).__init__()
        self.conv1 = svdconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = svdconv3x3(planes, planes)
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


class SVDBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SVDBottleneck, self).__init__()
        self.conv1 = SVDConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SVDConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SVDConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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


class SNResNet(nn.Module):

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10):
        super(SNResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SVDBottleneck if depth >=44 else SVDBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = SVDConv2d(3, self.inplanes, kernel_size=3, padding=1,
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
                SVDConv2d(self.inplanes, planes * block.expansion,
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
            if isinstance(m, SVDConv2d):
                reg += m.orth_reg()
        return reg

    def spectral_reg(self):
        reg = 0
        for m in self.modules():
            if isinstance(m, SVDConv2d):
                reg += m.spectral_reg()
        return reg

    def project(self):
        for m in self.modules():
            if isinstance(m, SVDConv2d):
                m.project()

    def showOrthInfo(self):
        for m in self.modules():
            if isinstance(m, SVDConv2d):
                m.showOrthInfo()


def svd_resnet(**kwargs):
    """
    Constructs a SNResNet model.
    """
    return SNResNet(**kwargs)

def svd_resnet110(**kwargs):
    """Constructs a SNResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(110, **kwargs)
    return model

def svd_resnet56(**kwargs):
    """Constructs a SNResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(56, **kwargs)
    return model

def svd_resnet44(**kwargs):
    """Constructs a SNResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(44, **kwargs)
    return model

def svd_resnet32(**kwargs):
    """Constructs a SNResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(32, **kwargs)
    return model

def svd_resnet20(**kwargs):
    """Constructs a SNResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(20, **kwargs)
    return model

def svd_wresnet20(**kwargs):
    """Constructs a Wide SNResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def svd_wresnet32(**kwargs):
    """Constructs a Wide SNResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(32, nfilter = [64,128,256], **kwargs)
    return model

def svd_wresnet44(**kwargs):
    """Constructs a Wide SNResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def svd_wresnet110(**kwargs):
    """Constructs a Wide SNResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SNResNet(110, nfilter = [64,128,256], **kwargs)
    return model
