from __future__ import absolute_import

'''Resnet with Simplex Convolution Laer
'''
import torch.nn as nn
import math
from ..SimplexConv import Simplex_Conv2d


__all__ = ['simplex_resnet','simplex_resnet110','simplex_resnet56','simplex_resnet44',
        'simplex_resnet32','simplex_resnet20','simplex_wresnet20','simplex_wresnet44','simplex_wresnet110']

def simplex_conv3x3(in_planes, out_planes, stride=1):
    "simplex 3x3 convolution with padding"
    return Simplex_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SimplexBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SimplexBasicBlock, self).__init__()
        self.sconv1 = simplex_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sconv2 = simplex_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.sconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimplexBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SimplexBottleneck, self).__init__()
        self.sconv1 = Simplex_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sconv2 = Simplex_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sconv3 = Simplex_Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.sconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.sconv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimplexResNet(nn.Module):

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10):
        super(SimplexResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SimplexBottleneck if depth >=44 else SimplexBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
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
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
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
            if isinstance(m, Simplex_Conv2d):
                m.project()


def simplex_resnet(**kwargs):
    """
    Constructs a SimplexResNet model.
    """
    return SimplexResNet(**kwargs)

def simplex_resnet110(**kwargs):
    """Constructs a SimplexResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(110, **kwargs)
    return model

def simplex_resnet56(**kwargs):
    """Constructs a SimplexResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(56, **kwargs)
    return model

def simplex_resnet44(**kwargs):
    """Constructs a SimplexResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(44, **kwargs)
    return model

def simplex_resnet32(**kwargs):
    """Constructs a SimplexResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(32, **kwargs)
    return model

def simplex_resnet20(**kwargs):
    """Constructs a SimplexResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(20, **kwargs)
    return model

def simplex_wresnet20(**kwargs):
    """Constructs a Wide SimplexResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def simplex_wresnet44(**kwargs):
    """Constructs a Wide SimplexResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def simplex_wresnet110(**kwargs):
    """Constructs a Wide SimplexResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SimplexResNet(110, nfilter = [64,128,256], **kwargs)
    return model
