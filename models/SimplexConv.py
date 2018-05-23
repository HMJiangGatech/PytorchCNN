'''
Usage
'''
import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from torch.autograd import Variable
import math

__all__ = ['Simplex_Conv2d']

class Simplex_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False, simplex_proj_algo = 'naive'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Simplex_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))

        # define auxilary parameter for simplex_proj_algo
        if(simplex_proj_algo == 'naive'):
            self.countcolnum = torch.ones(out_channels, kernel_size[0] * kernel_size[1] * in_channels).cumsum(dim=1)

        self.project()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def project(self, sparsity = 0.7, algo = 'naive'):
        '''
        Project weight to simplex

        Naive Algorithm ('naive'):
        Algo1 in https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        '''
        outputSize = self.weight.data.size()[0]
        _weight = self.weight.data.view(self.out_channels,-1)
        self.weight.data =  self.weight.data/ torch.norm(_weight,2,1).clamp(min = 1e-8).view(-1,1,1,1)
        if algo == 'naive':
            abs_weight = _weight.abs()
            z = sparsity*torch.ones(self.out_channels,1)

            sorted_abswei, _ = abs_weight.sort(dim=1,descending=True)
            theta = (sorted_abswei.cumsum(dim=1)-z)/self.countcolnum
            rho = (((sorted_abswei - theta)>0).sum(dim=1).long()-1).view(-1,1)
            thre = theta.gather(dim=1,index=rho)

            _weight[_weight<=thre] = 0
            _weight += -(_weight>thre).float()*thre + (_weight<-thre).float()*thre

        else:
            raise RuntimeError(algo+" projection not implemented yet")
