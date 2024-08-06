import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import Bottleneck, BasicBlock
from torch.nn.modules.container import Sequential


class bat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        x_int = torch.floor(0.5*(x/alpha+1)+torch.rand_like(x)).clamp(0, 1)
        output = (2*x_int-1)*alpha
        ctx.save_for_backward(x, x_int)
        ctx.other = alpha
        return output

    @staticmethod
    def backward(ctx, dy):
        x, x_int = ctx.saved_tensors
        alpha = ctx.other
        x_fp = x / alpha
        small = (x_fp < -1).float()
        big = (x_fp > 1).float()
        middle = 1.0 - small - big 
        dx = dy * middle
        dalpha = ((big-small+middle*(2*x_int-(x_fp+1)))*dy).sum()
        return dx, dalpha

class BAT_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BAT_Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.zeros((self.out_features, self.in_features)), requires_grad=False)
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.alpha0 = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.mask = torch.tensor(0.0)
        
    def set_parameters(self, rho):
        self.update.data = torch.zeros_like(self.weight)
        self.alpha.data = torch.tensor(0.0)
        self.rho.data = torch.tensor(float(rho))

    def init_alpha0(self):
        self.alpha0.data = self.update.abs().mean()
        
    def get_alpha(self):
        with torch.no_grad():
            alpha = self.alpha0*torch.exp(self.rho*self.alpha)
            alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
            return alpha

    def get_weight(self):
        with torch.no_grad():
            alpha = self.alpha0*torch.exp(self.rho*self.alpha)
            alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
            weight = self.weight + bat.apply(self.update, alpha)
            return weight

    def forward(self, x):
        alpha = self.alpha0*torch.exp(self.rho*self.alpha)
        alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
        update = bat.apply(self.update, alpha)
        update = update * self.mask + self.update * (1-self.mask)
        return F.linear(x, self.weight+update, None)

class BAT_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BAT_Conv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)), requires_grad=False)
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.alpha0 = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.mask = torch.tensor(0.0)
        
    def set_parameters(self, rho):
        self.update.data = torch.zeros_like(self.weight)
        self.alpha.data = torch.tensor(0.0)
        self.rho.data = torch.tensor(float(rho))

    def init_alpha0(self):
        self.alpha0.data = self.update.abs().mean()
        
    def get_alpha(self):
        with torch.no_grad():
            alpha = self.alpha0*torch.exp(self.rho*self.alpha)
            alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
            return alpha

    def get_weight(self):
        with torch.no_grad():
            alpha = self.alpha0*torch.exp(self.rho*self.alpha)
            alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
            weight = self.weight + bat.apply(self.update, alpha)
            return weight

    def forward(self, x):
        alpha = self.alpha0*torch.exp(self.rho*self.alpha)
        alpha = torch.clamp(alpha, torch.tensor(5e-4).to(alpha.device), torch.tensor(5e-2).to(alpha.device))
        update = bat.apply(self.update, alpha)
        update = update * self.mask + self.update * (1-self.mask)
        return F.conv2d(x, self.weight+update, None, self.stride, self.padding)


def bat_replace_modules(model):
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, BAT_Conv2d(module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.padding))
        if isinstance(module, nn.Linear):
            setattr(model, name, BAT_Linear(module.in_features, module.out_features))
        if isinstance(module, (Sequential, Bottleneck, BasicBlock)):
            bat_replace_modules(module)

def bat_set_parameters(model, rho):
    for name, module in model._modules.items():
        if isinstance(module, (BAT_Linear, BAT_Conv2d)):
            module.set_parameters(rho)    
        if isinstance(module, (Sequential, Bottleneck, BasicBlock)):
            bat_set_parameters(module, rho)

def bat_set_alpha0(model):
    for name, module in model._modules.items():
        if isinstance(module, (BAT_Linear, BAT_Conv2d)):
            module.init_alpha0()
        if isinstance(module, (Sequential, Bottleneck, BasicBlock)):
            bat_set_alpha0(module)

def bat_set_mask(model, mask):
    for name, module in model._modules.items():
        if isinstance(module, (BAT_Linear, BAT_Conv2d)):
            module.mask = torch.tensor(float(mask)) 
        if isinstance(module, (Sequential, Bottleneck, BasicBlock)):
            bat_set_mask(module, mask)

def bat_get_parameters(model):
    for name, module in model._modules.items():
        if isinstance(module, (BAT_Linear, BAT_Conv2d)):
            module.weight.data = module.get_weight()
        if isinstance(module, (Sequential, Bottleneck, BasicBlock)):
            bat_get_parameters(module)


class bat_solver():
    def __init__(self, model, total_sep, phi=0.5):
        self.total_sep = total_sep
        self.phi = phi
        self.iter = 0
        self.init = 0
        bat_set_mask(model, 0)
        
    def step(self, model):
        self.iter += 1
        if self.iter > self.phi*self.total_sep:
            if self.init == 0:
                bat_set_alpha0(model)
                bat_set_mask(model, 1)
                self.init = 1

