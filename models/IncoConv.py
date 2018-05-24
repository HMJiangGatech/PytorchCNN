# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math

__all__ = ["IncoConv2d"]

class IncoConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IncoConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.register_buffer('Im',torch.eye(out_channels)+(math.log(out_channels)/(in_channels*kernel_size[0]*kernel_size[1]))**0.5)

        self.scale = Parameter(torch.Tensor(1))
        self.scale.data.fill_(1)

        self.weight.data.normal_(0,0.02)
        if bias:
            self.bias.data.fill_(0)

    @property
    def W_(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8)
        return self.weight * self.scale

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def project(self):
        pass

    def showOrthInfo(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        W = self.weight.data.view(outputSize,-1)
        _, s, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max().item())
        print('mean:',s.mean().item())
        print('min :',s.min().item())
        print('var :',s.var().item())
        return s


    def orth_reg(self):
        outputSize = self.weight.data.size()[0]
        W = self.weight.view(outputSize,-1)
        return ((F.relu(W.mm(W.t())-self.Im))**2).sum()
