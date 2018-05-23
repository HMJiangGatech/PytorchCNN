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
import math

"""
orth_style in SVD or Householder or Cayley

Cayley:
https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
SVD:
https://arxiv.org/pdf/1709.06079.pdf
Householder:
https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
"""
__all__ = ['Orth_Plane_Conv2d','Orth_SVD_Conv2d','Orth_HH_Conv2d']

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

        self.eps = 1e-8
        self.norm = norm
        self.w_norm = w_norm
        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.projectiter = 0
        self.project(style='svd', interval = 1)

    def forward(self, input):

        _weight = self.weight
        _input = input
        # if self.w_norm:
        #     _weight = _weight/ torch.norm(_weight.view(self.out_channels,-1),2,1).clamp(min = self.eps).view(-1,1,1,1)

        _output = F.conv2d(input, _weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(_input**2, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

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
        print('penalty :', (W.mm(W.t())-torch.eye(outputSize)).norm()**2  )


class MySVD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        u,s,v = torch.svd(x)
        ctx.save_for_backward(x,u,s,v)
        return u,s,v

    @staticmethod
    def backward(ctx, gu, gs, gv):
        """
        https://j-towns.github.io/papers/svd-derivative.pdf

        This makes no assumption on the signs of sigma.
        """
        x,u,s,v = ctx.saved_tensors

        u = Variable(u)
        v = Variable(v)
        x = Variable(x)
        s = Variable(s)

        vt = v.t()
        ut = u.t()

        m = x.size()[0]
        n = x.size()[1]
        k = s.size()[0]

        sigma_mat = s.diag()
        sigma_mat_inv = (1/s).diag()
        sigma_expanded_sq = s.pow(2).expand_as(sigma_mat)
        Im = Variable(torch.eye(m))
        In = Variable(torch.eye(n))
        gvt = gv.t()
        f = sigma_expanded_sq - sigma_expanded_sq.t() + Variable(torch.eye(k))
        f = (f**-1) - Variable(torch.eye(k))

        sigma_term = (u.mm(gs.diag())).mm(vt)

        u_term = u.mm(f.mul(ut.mm(gu) - gu.t().mm(u))).mm(sigma_mat)
        if (m > k):
          u_term = u_term + (Im - u.mm(ut)).mm(gu).mm(sigma_mat_inv)
        u_term = u_term.mm(vt);

        v_term = sigma_mat.mm(f.mul(vt.mm(gv) - gvt.mm(v))).mm(vt)
        if (n > k):
          v_term = v_term + sigma_mat_inv.mm(gvt.mm(In - v.mm(vt)))
        v_term = u.mm(v_term);

        return u_term + sigma_term + v_term

svd = MySVD.apply

class Orth_SVD_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,
                norm=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_SVD_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.eps = 1e-8
        self.norm = norm
        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.weight_col_size = self.kernel_size[0] * self.kernel_size[1] * self.out_channels

    def forward(self, input):

        # V => W
        originSize = self.weight.size()
        outputSize = originSize[0]
        V = self.weight.view(outputSize, -1)
        Vc = V-V.mean(dim=1).view(-1,1)
        D, _, U = svd(Vc)
        W = (D.mm(U.t())).view(originSize)

        _output = F.conv2d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(input*input, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output


    def project(self):
        pass

class MyHH(torch.autograd.Function):

    @staticmethod
    def forward(ctx, V):
        n = V.size()[0]
        m = V.size()[1]
        W = torch.eye(n,m)
        Im = torch.eye(m)
        for i in range(n):
            u = V[i,:]
            W = W.mm(Im-2*u.view(-1,1).mm(u.view(1,-1)))
        ctx.save_for_backward(V,W)
        return W

    @staticmethod
    def backward(ctx, gradW):
        V,W = ctx.saved_variables
        n = gradW.size()[0]
        m = gradW.size()[1]
        gradV = Variable(torch.zeros(n,m))
        gradH = W.t().mm(gradW)
        Im = Variable(torch.eye(m))
        H = Variable(torch.eye(m))
        for i in range(n-1,-1,-1):
            u = V[i,:]
            gradH = gradH.mm(H)
            H = Im-2*u.view(-1,1).mm(u.view(1,-1))
            gradH = H.mm(gradH)
            du = -2*(torch.mv(gradH,u) + torch.mv(gradH.t(),u))
            gradV[i,:] = du
        return gradV

householder = MyHH.apply

class Orth_HH_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,
                norm=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_HH_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.eps = 1e-8
        self.norm = norm
        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.weight_col_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels

    def forward(self, input):

        # V => W
        originSize = self.weight.size()
        outputSize = originSize[0]
        V = self.weight.view(outputSize, -1)
        W = householder(V)
        W = W.view(originSize)

        _output = F.conv2d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(input*input, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output

    def project(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)

class Orth_AproxExp_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,
                norm=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_AproxExp_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.eps = 1e-8
        self.norm = norm
        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.weight_col_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels

    def forward(self, input):

        # V => W
        originSize = self.weight.size()
        outputSize = originSize[0]
        V = self.weight.view(outputSize, -1)
        W = householder(V)
        W = W.view(originSize)

        _output = F.conv2d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(input*input, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output

    def project(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)
