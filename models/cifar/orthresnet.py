from __future__ import absolute_import

'''ShpereNet
'''
import torch
import torch.nn as nn
import numpy as np
import math
from ..OrthConv import *

Orth_Conv2d = Orth_Plane_Conv2d

__all__ = ['orth_resnet','orth_resnet110','orth_resnet56','orth_resnet44',
           'orth_resnet32','orth_resnet20','orth_wresnet20','orth_wresnet32',
           'orth_wresnet44', 'orth_wresnet110']

def orth_conv3x3(in_planes, out_planes, stride=1):
    "orth 3x3 convolution with padding"
    return Orth_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class GroupOrthConv(nn.Module):
    '''
    devide output channels into 'groups'
    '''
    def __init__(self, in_channels, out_channels, groups=None, kernel_size=3,
                stride=1, padding=0, bias=False):
        super(GroupOrthConv, self).__init__()
        if groups == None:
            groups = (out_channels-1)//(in_channels*kernel_size*kernel_size)+1
        self.groups = groups
        self.gourp_out_channels = np.ones(groups) * (out_channels//groups)
        if out_channels%groups > 0:
            self.gourp_out_channels[:out_channels%groups] += 1
        self.sconvs = []
        for i in range(groups):
            newsconv = Orth_Conv2d(in_channels, int(self.gourp_out_channels[i]),
                            kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)
            self.add_module('sconv_'+str(i),newsconv)
            self.sconvs.append(newsconv)

    def forward(self,x):
        out = []
        for i in range(self.groups):
            out.append(self.sconvs[i](x))
        return torch.cat(out,1)


class OrthBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(OrthBasicBlock, self).__init__()
        self.sconv1 = orth_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sconv2 = orth_conv3x3(planes, planes)
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


class OrthBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(OrthBottleneck, self).__init__()
        self.sconv1 = Orth_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sconv2 = Orth_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sconv3 = Orth_Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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


class OrthResNet(nn.Module):

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10, rescale_bn=False):
        super(OrthResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        self.depth = n*3

        block = OrthBottleneck if depth >=44 else OrthBasicBlock

        self.inplanes = nfilter[0]
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
        #                        bias=False)
        self.sconv1 = GroupOrthConv(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nfilter[0], n)
        self.layer2 = self._make_layer(block, nfilter[1], n, stride=2)
        self.layer3 = self._make_layer(block, nfilter[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nfilter[2] * block.expansion, num_classes)

        self.features = nn.Sequential(
                self.sconv1,
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

    def orth_reg(self):
        reg = 0
        for m in self.modules():
            if isinstance(m, Orth_Conv2d):
                reg += m.orth_reg()
        return reg

    def project(self):
        for m in self.modules():
            if isinstance(m, Orth_Conv2d):
                m.project()

    def showOrthInfo(self):
        for m in self.modules():
            if isinstance(m, Orth_Conv2d):
                m.showOrthInfo()


def orth_resnet(**kwargs):
    """
    Constructs a OrthResNet model.
    """
    return OrthResNet(**kwargs)

def orth_resnet110(**kwargs):
    """Constructs a OrthResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(110, **kwargs)
    return model

def orth_resnet56(**kwargs):
    """Constructs a OrthResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(56, **kwargs)
    return model

def orth_resnet44(**kwargs):
    """Constructs a OrthResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(44, **kwargs)
    return model

def orth_resnet32(**kwargs):
    """Constructs a OrthResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(32, **kwargs)
    return model

def orth_resnet20(**kwargs):
    """Constructs a OrthResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(20, **kwargs)
    return model

def orth_wresnet20(**kwargs):
    """Constructs a Wide OrthResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def orth_wresnet32(**kwargs):
    """Constructs a Wide OrthResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(32, nfilter = [64,128,256], **kwargs)
    return model

def orth_wresnet44(**kwargs):
    """Constructs a Wide OrthResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def orth_wresnet110(**kwargs):
    """Constructs a Wide OrthResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrthResNet(110, nfilter = [64,128,256], **kwargs)
    return model
