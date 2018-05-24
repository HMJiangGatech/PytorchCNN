'''
Usage
'''
import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules import Module
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

__all__ = ['Orth_Plane_Conv2d']

class Orth_Plane_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,
                norm=False, w_norm=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_Plane_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.scale = Parameter(torch.Tensor(1))
        self.scale.data.fill_(1)

        self.register_buffer('Im',torch.eye(out_channels))

        self.eps = 1e-8
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.projectiter = 0
        self.project(style='qr', interval = 1)

    def forward(self, input):
        _output = F.conv2d(input, self.weight*self.scale, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return _output

    def project(self, style='none', interval = 1):
        '''
        Project weight to l2 ball
        '''
        self.projectiter = self.projectiter+1
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        if style=='qr' and self.projectiter%interval == 0:
            # Compute the qr factorization
            q, r = torch.qr(self.weight.data.view(outputSize,-1).t())
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            self.weight.data = q.t().view(originSize)
        elif style=='svd' and self.projectiter%interval == 0:
            # Compute the svd factorization (may be not stable)
            u, s, v = torch.svd(self.weight.data.view(outputSize,-1))
            self.weight.data = u.mm(v.t()).view(originSize)
        elif self.w_norm:
            self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)

    def showOrthInfo(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        W = self.weight.data.view(outputSize,-1)
        _, s, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max())
        print('mean:',s.mean())
        print('min :',s.min())
        print('var :',s.var())
        print('penalty :', (W.mm(W.t())-self.Im).norm()**2  )

    def orth_reg(self):
        outputSize = self.weight.data.size()[0]
        W = self.weight.view(outputSize,-1)
        return ((W.mm(W.t())-self.Im)**2).sum()
