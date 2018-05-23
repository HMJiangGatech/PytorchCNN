import torch
from torch.optim.optimizer import Optimizer, required

class SMGD(Optimizer):
    r"""Implements stochastic manifold gradient descent with orthangonal constraint (optionally with momentum).
    Author: Feng Liu
    Fomular: manifold grad = (I-WW)( grad(L) + reg_weight * lambda * W(Wt*W - I))
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, reg_weight=1e-5, names=None, project_norm = False, manifold_grad = False):
        if names is None:
            names = []
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, reg_weight=reg_weight, names=names)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SMGD, self).__init__(params, defaults)
        self.project_norm = project_norm
        self.manifold_grad = manifold_grad

    def __setstate__(self, state):
        super(SMGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print('######### Step Separator############')
        # use_cuda = torch.cuda.is_available()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            reg_weight = group['reg_weight']
            names = group['names']

            print(names)

            convLayer = [i for i in range(len(names)) if 'conv' in names[i]]

            # print('######### Group Separator############')

            for i in range(len(group['params'])):

                p = group['params'][i]

                if p.grad is None:
                    continue
                # print('######### p Separator############')
                if 'conv' in names[i]:
                    # veloc = (I - WWt) * veloc
                    originSize = p.data.size()
                    outputSize = p.data.size()[0]

                    Wt = p.data.view(outputSize, -1)
                    W = torch.t(Wt)
                    WWt = W.mm(Wt)
                    WtW = Wt.mm(W)

                    if self.manifold_grad and 'layer' in names[i]:
                        # version 1 (GWT-WGT)W
                        # I = torch.eye(WWt.size()[0], WWt.size()[1])
                        # buf = I.sub(WWt)
                        # d_p = p.grad.data.view(outputSize, -1)
                        # d_p = d_p.mm(buf)
                        # d_p = d_p.view(originSize)

                        # version 2 d_p = (GWt-WGt)W
                        Gt = p.grad.data.view(outputSize, -1)
                        G = torch.t(Gt)
                        GWt = G.mm(Wt)

                        WGt = W.mm(Gt)
                        A = GWt.sub(WGt)
                        d_p = Wt.mm(torch.t(A))
                        d_p = d_p.view(originSize)
                    else:
                        # gradW = Original d_p
                        gradW = p.grad.data.view(outputSize, -1)
                        # shift = lambda * (WWt-I)W
                        I = torch.eye(WtW.size()[0], WtW.size()[1])
                        shift = torch.t(W.mm(WtW.sub(I)))

                        # veloc = buf * (gradW + shift)
                        d_p = gradW.add_(reg_weight, shift)
                        d_p = d_p.view(originSize)
                else:
                    d_p = p.grad.data
                    # add weight decay
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                # adjust with momentum
                # veloc = mom * veloc + grad
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # W = W - lr * veloc
                p.data.add_(-group['lr'], d_p)

            # W_conv = W_conv / (W_conv.norm)
            if self.project_norm:
                for i in convLayer:
                    p = group['params'][i]
                    outputSize = p.data.size()[0]
                    p.data =  p.data/ torch.norm(p.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)

        return loss
