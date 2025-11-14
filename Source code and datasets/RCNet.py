#Real Part
import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel // 4)
        self.conv2 = nn.Conv2d(output_channel // 4, output_channel // 4, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel // 4)
        self.conv3 = nn.Conv2d(output_channel // 4, output_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False)
        self.bn4 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        residual = self.bn4(self.conv4(residual))
        out = x + residual
        out = self.relu(out)
        return out
class IdentityBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel // 4)
        self.conv2 = nn.Conv2d(output_channel // 4, output_channel // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel // 4)
        self.conv3 = nn.Conv2d(output_channel // 4, output_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.se=CBAMBlock(output_channel)
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x=self.se(x)
        out = x + residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self,  num_classes=7):#ConvBlock, IdentityBlock
        super(ResNet50, self).__init__()
        self.inchannel = 64
        self.Conv1 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Layer1 = nn.Sequential(
            ConvBlock(20, 20, 1),
            # IdentityBlock(20, 20),
            IdentityBlock(20, 20)
        )
        self.Layer2 = nn.Sequential(
            ConvBlock(20, 40, 2),
            # IdentityBlock(40, 40),
            # IdentityBlock(40, 40),
            IdentityBlock(40, 40)
        )
        self.Layer3 = nn.Sequential(
            ConvBlock(40, 80, 2),
            # IdentityBlock(80, 80),
            # IdentityBlock(80, 80),
            # IdentityBlock(80, 80),
            # IdentityBlock(80, 80),
            IdentityBlock(80, 80)
        )
        self.Layer4 = nn.Sequential(
            ConvBlock(80,128, 2),
            # IdentityBlock(128, 128),
            IdentityBlock(128, 128)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):  # 3*224*224
        out = self.Conv1(x)  # 64*56*56
        out = self.Layer1(out)  # 256*56*56
        out = self.Layer2(out)  # 512*28*28
        out = self.Layer3(out)  # 1024*14*14
        out = self.Layer4(out)  # Tensor:(200，1024，4，4)
        out = self.avg_pool(out)  # 2048*1*1
        return out
#complex part
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io
import math
import random



# import time

# ---------------------Functions----------------------------

def seeds_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  ##  CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Utility functions for initialization
def _istuple(x):   return isinstance(x, tuple)


def _mktuple1d(x): return x if _istuple(x) else (x,)


def _mktuple2d(x): return x if _istuple(x) else (x, x)


def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    if not fanin:
        fanin = 1
        for p in Wi.shape[1:]: fanin *= p
    scale = float(gain) / float(fanin)
    theta = torch.empty_like(Wr).uniform_(-math.pi / 2, +math.pi / 2)
    rho = np.random.rayleigh(scale, tuple(Wr.shape))
    rho = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho * theta.cos())
    Wi.data.copy_(rho * theta.sin())


def complex_relu(input_r, input_i):
    output_r = relu(input_r)
    output_i = relu(input_i)
    return output_r, output_i


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output


def complex_max_pool2d(input_r, input_i, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    complex_abs = torch.sqrt(torch.pow(input_r, 2) + torch.pow(input_i, 2))
    absolute_value, indices = max_pool2d(
        complex_abs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented

    # angle = torch.atan2(input_i,input_r)
    # get only the phase values selected by max pool

    # angle = _retrieve_elements_from_indices(angle, indices)
    output_r = _retrieve_elements_from_indices(input_r, indices)
    output_i = _retrieve_elements_from_indices(input_i, indices)

    return output_r, output_i
    # return absolute_value \
    # * (torch.cos(angle).type(torch.complex64)+1j*torch.sin(angle).type(torch.complex64))


def complex_avg_pool2d(input_r, input_i, kernel_size, stride=None, padding=0):
    output_r = avg_pool2d(input_r, kernel_size=kernel_size, stride=stride, padding=padding)
    output_i = avg_pool2d(input_i, kernel_size=kernel_size, stride=stride, padding=padding)

    return output_r, output_i



def mag_max_fusion(A_r, B_r, C_r, A_i, B_i, C_i):
    # complex magnitude
    A_abs = torch.sqrt(torch.pow(A_r, 2) + torch.pow(A_i, 2))
    B_abs = torch.sqrt(torch.pow(B_r, 2) + torch.pow(B_i, 2))
    C_abs = torch.sqrt(torch.pow(C_r, 2) + torch.pow(C_i, 2))

    m, n, p, q = A_abs.size()

    magnitude = torch.stack([A_abs.view(-1), B_abs.view(-1), C_abs.view(-1)], dim=0)
    am = torch.stack([A_r.view(-1), B_r.view(-1), C_r.view(-1)], dim=0).flatten()
    ph = torch.stack([A_i.view(-1), B_i.view(-1), C_i.view(-1)], dim=0).flatten()

    # find maximum magnitude values and indices
    mag_max_indices = torch.max(magnitude, 0)[1] 

    # retrive the max values from real and imaginary parts
    # bias = torch.arange(start=0, end=m*n*p*q, step=1).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    bias = torch.arange(start=0, end=m * n * p * q, step=1).cuda()
    index = mag_max_indices * m * n * p * q + bias  

    output_r = am.gather(dim=0, index=index).view_as(A_abs) 
    output_i = ph.gather(dim=0, index=index).view_as(A_abs)

    return output_r, output_i


'''    
    am_max = torch.Tensor(m*n*p*q).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    ph_max = torch.Tensor(m*n*p*q).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    for i in range(m*n*p*q):
        am_max[i] = am[mag_max_indice[i], i]
        ph_max[i] = ph[mag_max_indice[i], i]
    output_r = am_max.reshape(m,n,p,q)
    output_i = ph_max.reshape(m,n,p,q)

    return output_r, output_i
'''



def am_max_fusion(A_r, B_r, C_r, A_i, B_i, C_i):
    # faltten and merge togrther
    am = torch.stack([A_r.view(-1), B_r.view(-1), C_r.view(-1)], dim=0)
    ph = torch.stack([A_i.view(-1), B_i.view(-1), C_i.view(-1)], dim=0).flatten()

    # find maximum am values and indices
    am_max = torch.max(am, 0)
    am_max_values = am_max.values  
    am_max_indices = am_max.indices  

    # retrive the max values from real and imaginary parts
    bias = torch.arange(start=0, end=am.size()[1], step=1).cuda()
    index = am_max_indices * am.size()[1] + bias 

    output_r = am_max_values.view_as(A_r) 
    output_i = ph.gather(dim=0, index=index).view_as(A_r)

    return output_r, output_i


# -----------------------Layers-----------------------------
#              Feature Fusion
class Avg_Fusion(Module):

    def forward(self, Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i):
        output_r, output_i = (Fea_A_r + Fea_B_r + Fea_C_r) / 3, (Fea_A_i + Fea_B_i + Fea_C_i) / 3
        return output_r, output_i


class Max_Fusion(Module):

    def forward(self, Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i):
        # start = time.time()
        #   amplitude maximum fusuin - AMF
        # output_r, output_i = am_max_fusion(Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i)
        #   magnitude maximum fusion - MMF
        output_r, output_i = mag_max_fusion(Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i)
        # print('output size:', output_r.size())
        # end = time.time()
        # print('time consumption:', end-start)
        return output_r, output_i


#
class Ada_Fusion(Module):
    def __init__(self, num):
        super(Ada_Fusion, self).__init__()
        self.weight_Ar = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Br = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Cr = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Ai = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Bi = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Ci = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_Ar.data.fill_(1 / num)
        self.weight_Br.data.fill_(1 / num)
        self.weight_Cr.data.fill_(1 / num)
        self.weight_Ai.data.fill_(1 / num)
        self.weight_Bi.data.fill_(1 / num)
        self.weight_Ci.data.fill_(1 / num)

    def forward(self, Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i):
        output_r = self.weight_Ar * Fea_A_r + self.weight_Br * Fea_B_r + self.weight_Cr * Fea_C_r
        output_i = self.weight_Ai * Fea_A_i + self.weight_Bi * Fea_B_i + self.weight_Ci * Fea_C_i
        return output_r, output_i


class Concat_Fusion(Module):
    def forward(self, Fea_A_r, Fea_B_r, Fea_C_r, Fea_A_i, Fea_B_i, Fea_C_i):
        output_r = torch.cat((Fea_A_r, Fea_B_r, Fea_C_r), 1)
        output_i = torch.cat((Fea_A_i, Fea_B_i, Fea_C_i), 1)
        return output_r, output_i


#     ComplexReLU()
class ComplexReLU(Module):

    def forward(self, input_r, input_i):
        output_r, output_i = complex_relu(input_r, input_i)
        return output_r, output_i


#      ComplexMaxPool2d
class ComplexMaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input_r, input_i):
        return complex_max_pool2d(input_r, input_i, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)


#      ComplexAvgPool2d
class ComplexAvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input_r, input_i):
        return complex_avg_pool2d(input_r, input_i, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding)


##     ComplexConv2d
class ComplexConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)
        self.groups = groups

        self.Wr = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        self.Wi = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        if bias:
            self.Br = torch.nn.Parameter(torch.Tensor(out_channels))
            self.Bi = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)
        self.reset_parameters()

    def reset_parameters(self):
        fanin = self.in_channels // self.groups  
        for s in self.kernel_size:
            fanin *= s
        complex_rayleigh_init(self.Wr, self.Wi, fanin)  ##  complex_rayleigh_init
        if self.Br is not None and self.Bi is not None:
            self.Br.data.zero_()
            self.Bi.data.zero_()

    def forward(self, xr, xi):
        yrr = F.conv2d(xr, self.Wr, self.Br, self.stride, self.padding, self.dilation, self.groups)
        yri = F.conv2d(xr, self.Wi, self.Bi, self.stride, self.padding, self.dilation, self.groups)
        yir = F.conv2d(xi, self.Wr, None, self.stride, self.padding, self.dilation, self.groups)
        yii = F.conv2d(xi, self.Wi, None, self.stride, self.padding, self.dilation, self.groups)
        return yrr - yii, yri + yir



class ComplexBatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = Parameter(torch.Tensor(num_features))
            self.Wri = Parameter(torch.Tensor(num_features))
            self.Wii = Parameter(torch.Tensor(num_features))
            self.Br = Parameter(torch.Tensor(num_features))
            self.Bi = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(num_features))
            self.register_buffer('RMi', torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones(num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        if training:
            Mr = xr
            Mi = xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        # Matrix Inverse Square Root U = V^-0.5
        tau = Vrr + Vii
        delta = Vrr * Vii - Vri.pow(2)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()
        rst = (s * t).reciprocal()

        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        if self.affine:
            Zrr = self.Wrr[None, :, None, None] * Urr + self.Wri[None, :, None, None] * Uri
            Zri = self.Wrr[None, :, None, None] * Uri + self.Wri[None, :, None, None] * Uii
            Zir = self.Wri[None, :, None, None] * Urr + self.Wii[None, :, None, None] * Uri
            Zii = self.Wri[None, :, None, None] * Uri + self.Wii[None, :, None, None] * Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr, yi = Zrr * xr + Zri * xi, Zir * xr + Zii * xi
        if self.affine:
            yr = yr + self.Br[None, :, None, None]
            yi = yi + self.Bi[None, :, None, None]

        return yr, yi

        def extra_repr(self):
            return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                   'track_running_stats={track_running_stats}'.format(**self.__dict__)

        def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys,
                                  unexpected_keys, error_msgs):
            super(ComplexBatchNorm, self)._load_from_state_dict(state_dict,
                                                                prefix,
                                                                strict,
                                                                missing_keys,
                                                                unexpected_keys,
                                                                error_msgs)


##   Complex-Linear
class ComplexLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Wr = Parameter(torch.Tensor(out_features, in_features))
        self.Wi = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.Br = Parameter(torch.Tensor(out_features))
            self.Bi = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        self.reset_parameters()

    def reset_parameters(self):
        complex_rayleigh_init(self.Wr, self.Wi, self.in_features)
        if self.Br is not None and self.Bi is not None:
            self.Br.data.zero_()
            self.Bi.data.zero_()

    def forward(self, xr, xi):
        yrr = torch.nn.functional.linear(xr, self.Wr, self.Br)
        yri = torch.nn.functional.linear(xr, self.Wi, self.Bi)
        yir = torch.nn.functional.linear(xi, self.Wr, None)
        yii = torch.nn.functional.linear(xi, self.Wi, None)
        return yrr - yii, yri + yir


# ---------------------ComplexNet Architecture-----------------------
class MSCVNet(nn.Module):
    ''' this is the backbone network of Multi-Streams Complex Value Networks'''

    def __init__(self, num_classes):
        super(MSCVNet, self).__init__()
        # ----------------Stream_A-kernel_size=3----------------#
        # ----------------Conv_Layer1------------#
        self.Conv_A1 = ComplexConv2d(3, 40, kernel_size=3, stride=1, padding=1) 
        self.BN_A1 = ComplexBatchNorm(40)
        self.ReLU_A1 = ComplexReLU()
        # self.MaxPool2d_A1 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer2---------------------------
        self.Conv_A2 = ComplexConv2d(40, 40, kernel_size=3, stride=1, padding=1)
        self.BN_A2 = ComplexBatchNorm(40)
        self.ReLU_A2 = ComplexReLU()
        self.MaxPool2d_A2 = ComplexMaxPool2d(kernel_size=2, stride=2)

        # ----------------Stream_B-kernel_size=7----------------#
        # ----------------Conv_Layer1------------#
        self.Conv_B1 = ComplexConv2d(3, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B1 = ComplexBatchNorm(40)
        self.ReLU_B1 = ComplexReLU()
        # self.MaxPool2d_B1 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer2---------------------------
        self.Conv_B2 = ComplexConv2d(40, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B2 = ComplexBatchNorm(40)
        self.ReLU_B2 = ComplexReLU()
        self.MaxPool2d_B2 = ComplexMaxPool2d(kernel_size=2, stride=2)

        # ----------------Stream_C-kernel_size=11----------------#
        # ----------------Conv_Layer1------------#
        self.Conv_C1 = ComplexConv2d(3, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C1 = ComplexBatchNorm(40)
        self.ReLU_C1 = ComplexReLU()
        # self.MaxPool2d_C1 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer2---------------------------
        self.Conv_C2 = ComplexConv2d(40, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C2 = ComplexBatchNorm(40)
        self.ReLU_C2 = ComplexReLU()
        self.MaxPool2d_C2 = ComplexMaxPool2d(kernel_size=2, stride=2)

        # ---------------Fusion Layer1---------------------------------#
        # self.Avg_F1  = Avg_Fusion()
        # self.Max_F1  = Max_Fusion()
        # self.Ada_F1  = Ada_Fusion(3)
        self.Concat_F1 = Concat_Fusion()

        # ----------------Stream_A-kernel_size=3----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_A3 = ComplexConv2d(120, 40, kernel_size=3, stride=1, padding=1)
        self.BN_A3 = ComplexBatchNorm(40)
        self.ReLU_A3 = ComplexReLU()
        # self.MaxPool2d_A3 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_A4 = ComplexConv2d(40, 40, kernel_size=3, stride=1, padding=1)
        self.BN_A4 = ComplexBatchNorm(40)
        self.ReLU_A4 = ComplexReLU()
        #############################################################################
        self.MaxPool2d_A4 = ComplexMaxPool2d(kernel_size=1, stride=1)
        # self.AvgPool2d_A4 = ComplexAvgPool2d(kernel_size=2, stride=2)

        # ----------------Stream_B-kernel_size=5----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_B3 = ComplexConv2d(120, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B3 = ComplexBatchNorm(40)
        self.ReLU_B3 = ComplexReLU()
        # self.MaxPool2d_B3 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_B4 = ComplexConv2d(40, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B4 = ComplexBatchNorm(40)
        self.ReLU_B4 = ComplexReLU()
        #############################################################################
        self.MaxPool2d_B4 = ComplexMaxPool2d(kernel_size=1, stride=1)
        # self.AvgPool2d_B4 = ComplexAvgPool2d(kernel_size=2, stride=2)

        # ----------------Stream_C-kernel_size=7----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_C3 = ComplexConv2d(120, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C3 = ComplexBatchNorm(40)
        self.ReLU_C3 = ComplexReLU()
        # self.MaxPool2d_C3 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_C4 = ComplexConv2d(40, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C4 = ComplexBatchNorm(40)
        self.ReLU_C4 = ComplexReLU()
        #############################################################################
        self.MaxPool2d_C4 = ComplexMaxPool2d(kernel_size=1, stride=1)
        # self.AvgPool2d_C4 = ComplexAvgPool2d(kernel_size=2, stride=2)

        # ---------------Fusion Layer2---------------------------------#
        # self.Avg_F2  = Avg_Fusion()
        # self.Max_F2  = Max_Fusion()
        # self.Ada_F2  = Ada_Fusion(3)
        self.Concat_F2 = Concat_Fusion()

        # ----------------Stream_A-kernel_size=3----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_A5 = ComplexConv2d(120, 40, kernel_size=3, stride=1, padding=1)
        self.BN_A5 = ComplexBatchNorm(40)
        self.ReLU_A5 = ComplexReLU()
        self.MaxPool2d_A5 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_A6 = ComplexConv2d(40, 40, kernel_size=3, stride=1, padding=1)
        self.BN_A6 = ComplexBatchNorm(40)
        self.ReLU_A6 = ComplexReLU()
        # self.MaxPool2d_A6 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.AvgPool2d_A6 = ComplexAvgPool2d(kernel_size=1, stride=1)

        # ----------------Stream_B-kernel_size=5----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_B5 = ComplexConv2d(120, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B5 = ComplexBatchNorm(40)
        self.ReLU_B5 = ComplexReLU()
        self.MaxPool2d_B5 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_B6 = ComplexConv2d(40, 40, kernel_size=7, stride=1, padding=3)
        self.BN_B6 = ComplexBatchNorm(40)
        self.ReLU_B6 = ComplexReLU()
        # self.MaxPool2d_B6 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.AvgPool2d_B6 = ComplexAvgPool2d(kernel_size=1, stride=1)

        # ----------------Stream_C-kernel_size=7----------------#
        # --------------------------Conv_Layer3---------------------------
        self.Conv_C5 = ComplexConv2d(120, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C5 = ComplexBatchNorm(40)
        self.ReLU_C5 = ComplexReLU()
        self.MaxPool2d_C5 = ComplexMaxPool2d(kernel_size=2, stride=2)
        # --------------------------Conv_Layer4---------------------------
        self.Conv_C6 = ComplexConv2d(40, 40, kernel_size=11, stride=1, padding=5)
        self.BN_C6 = ComplexBatchNorm(40)
        self.ReLU_C6 = ComplexReLU()
        # self.MaxPool2d_C6 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.AvgPool2d_C6 = ComplexAvgPool2d(kernel_size=1, stride=1)

        # ---------------Fusion Layer3---------------------------------#
        # self.Avg_F3  = Avg_Fusion()
        # self.Max_F3  = Max_Fusion()
        # self.Ada_F3  = Ada_Fusion(3)
        self.Concat_F3 = Concat_Fusion()

        # ----------------Fusion Conv Layer--------------------------#
        self.Conv_Fu = ComplexConv2d(120, 128, kernel_size=1, stride=1, padding=0)
        self.BN_Fu = ComplexBatchNorm(128)
        self.ReLU_Fu = ComplexReLU()

        # ----------------Full Connection Layers----------------#
        self.FC1 = ComplexLinear(128, num_classes)
        self.ReLU_FC1 = ComplexReLU()

    def forward(self, X):
        # ----------------Stream_A-kernel_size=3----------------#
        # ----------------Layer1-------------------
        xr, xi = torch.chunk(X, 2, dim=1)
        xr_A, xi_A = self.Conv_A1(xr, xi)  # 64 x 64
        xr_A, xi_A = self.BN_A1(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A1(xr_A, xi_A)
        # xr_A, xi_A = self.MaxPool2d_A1(xr_A, xi_A)
        # ----------------Layer2-------------------
        xr_A, xi_A = self.Conv_A2(xr_A, xi_A)
        xr_A, xi_A = self.BN_A2(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A2(xr_A, xi_A)  # 64 x 64
        xr_A, xi_A = self.MaxPool2d_A2(xr_A, xi_A)  # 32 x 32

        # ----------------Stream_B-kernel_size=7----------------#
        # ----------------Layer1-------------------
        xr_B, xi_B = self.Conv_B1(xr, xi)  # 64 x 64
        xr_B, xi_B = self.BN_B1(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B1(xr_B, xi_B)
        # xr_B, xi_B = self.MaxPool2d_B1(xr_B, xi_B)
        # ----------------Layer2-------------------
        xr_B, xi_B = self.Conv_B2(xr_B, xi_B)
        xr_B, xi_B = self.BN_B2(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B2(xr_B, xi_B)
        xr_B, xi_B = self.MaxPool2d_B2(xr_B, xi_B)  # 32 x32

        # ----------------Stream_C-kernel_size=11----------------#
        # ----------------Layer1-------------------
        xr_C, xi_C = self.Conv_C1(xr, xi)  # 64 x 64
        xr_C, xi_C = self.BN_C1(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C1(xr_C, xi_C)
        # xr_C, xi_C = self.MaxPool2d_C1(xr_C, xi_C)
        # ----------------Layer2-------------------
        xr_C, xi_C = self.Conv_C2(xr_C, xi_C)
        xr_C, xi_C = self.BN_C2(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C2(xr_C, xi_C)
        xr_C, xi_C = self.MaxPool2d_C2(xr_C, xi_C)  # 32 x 32

        # ----------------Fusion Layer1--------------------------#
        # xr_F1, xi_F1 = self.Avg_F1(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)    # 32 x 32
        # xr_F1, xi_F1 = self.Max_F1(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        # xr_F1, xi_F1 = self.Ada_F1(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        xr_F1, xi_F1 = self.Concat_F1(xr_A, xr_B, xr_C, xi_A, xi_B, xi_C)

        # ----------------Stream_A-kernel_size=3----------------#
        # ----------------Layer3-------------------
        xr_A, xi_A = self.Conv_A3(xr_F1, xi_F1)  # 32 x 32
        xr_A, xi_A = self.BN_A3(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A3(xr_A, xi_A)
        # xr_A, xi_A = self.MaxPool2d_A3(xr_A, xi_A)
        # ----------------Layer4-------------------
        xr_A, xi_A = self.Conv_A4(xr_A, xi_A)
        xr_A, xi_A = self.BN_A4(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A4(xr_A, xi_A)
        xr_A, xi_A = self.MaxPool2d_A4(xr_A, xi_A)  # 16 x 16
        # xr_A, xi_A = self.AvgPool2d_A4(xr_A, xi_A)

        # ----------------Stream_B-kernel_size=7----------------#
        # ----------------Layer3-------------------
        xr_B, xi_B = self.Conv_B3(xr_F1, xi_F1)  # 32 x 32
        xr_B, xi_B = self.BN_B3(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B3(xr_B, xi_B)
        # xr_B, xi_B = self.MaxPool2d_B3(xr_B, xi_B)
        # ----------------Layer4-------------------
        xr_B, xi_B = self.Conv_B4(xr_B, xi_B)
        xr_B, xi_B = self.BN_B4(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B4(xr_B, xi_B)
        xr_B, xi_B = self.MaxPool2d_B4(xr_B, xi_B)  # 16 x 16
        # xr_B, xi_B = self.AvgPool2d_B4(xr_B, xi_B)

        # ----------------Stream_C-kernel_size=11----------------#
        # ----------------Layer3-------------------
        xr_C, xi_C = self.Conv_C3(xr_F1, xi_F1)  # 32 x 32
        xr_C, xi_C = self.BN_C3(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C3(xr_C, xi_C)
        # xr_C, xi_C = self.MaxPool2d_C3(xr_C, xi_C)
        # ----------------Layer4-------------------
        xr_C, xi_C = self.Conv_C4(xr_C, xi_C)
        xr_C, xi_C = self.BN_C4(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C4(xr_C, xi_C)
        xr_C, xi_C = self.MaxPool2d_C4(xr_C, xi_C)  # 16 x 16
        # xr_C, xi_C = self.AvgPool2d_C4(xr_C, xi_C)

        # ----------------Fusion Layer2--------------------------#
        # xr_F2, xi_F2 = self.Avg_F2(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        # xr_F2, xi_F2 = self.Max_F2(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        # xr_F2, xi_F2 = self.Ada_F2(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        xr_F2, xi_F2 = self.Concat_F2(xr_A, xr_B, xr_C, xi_A, xi_B, xi_C)

        # ----------------Stream_A-kernel_size=3----------------#
        # ----------------Layer5-------------------
        xr_A, xi_A = self.Conv_A5(xr_F2, xi_F2)  # 16 x 16
        xr_A, xi_A = self.BN_A5(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A5(xr_A, xi_A)
        xr_A, xi_A = self.MaxPool2d_A5(xr_A, xi_A)  # 8 x 8
        # ----------------Layer6-------------------
        xr_A, xi_A = self.Conv_A6(xr_A, xi_A)
        xr_A, xi_A = self.BN_A6(xr_A, xi_A)
        xr_A, xi_A = self.ReLU_A6(xr_A, xi_A)
        # xr_A, xi_A = self.MaxPool2d_A6(xr_A, xi_A)              # 4 x 4
        xr_A, xi_A = self.AvgPool2d_A6(xr_A, xi_A)

        # ----------------Stream_B-kernel_size=7----------------#
        # ----------------Layer5-------------------
        xr_B, xi_B = self.Conv_B5(xr_F2, xi_F2)  # 16 x 16
        xr_B, xi_B = self.BN_B5(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B5(xr_B, xi_B)
        xr_B, xi_B = self.MaxPool2d_B5(xr_B, xi_B)  # 8 x 8
        # ----------------Layer6-------------------
        xr_B, xi_B = self.Conv_B6(xr_B, xi_B)
        xr_B, xi_B = self.BN_B6(xr_B, xi_B)
        xr_B, xi_B = self.ReLU_B6(xr_B, xi_B)
        # xr_B, xi_B = self.MaxPool2d_B6(xr_B, xi_B)              # 4 x 4
        xr_B, xi_B = self.AvgPool2d_B6(xr_B, xi_B)

        # ----------------Stream_C-kernel_size=11----------------#
        # ----------------Layer5-------------------
        xr_C, xi_C = self.Conv_C5(xr_F2, xi_F2)  # 16 x 16
        xr_C, xi_C = self.BN_C5(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C5(xr_C, xi_C)
        xr_C, xi_C = self.MaxPool2d_C5(xr_C, xi_C)  # 8 x 8
        # ----------------Layer6-------------------
        xr_C, xi_C = self.Conv_C6(xr_C, xi_C)
        xr_C, xi_C = self.BN_C6(xr_C, xi_C)
        xr_C, xi_C = self.ReLU_C6(xr_C, xi_C)
        # xr_C, xi_C = self.MaxPool2d_C6(xr_C, xi_C)                # 4 x 4
        xr_C, xi_C = self.AvgPool2d_C6(xr_C, xi_C)

        # ----------------Fusion Layer3--------------------------#
        # xr_F3, xi_F3 = self.Avg_F3(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        # xr_F3, xi_F3 = self.Max_F3(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        # xr_F3, xi_F3 = self.Ada_F3(xr_A,xr_B,xr_C,xi_A,xi_B,xi_C)
        xr_F3, xi_F3 = self.Concat_F3(xr_A, xr_B, xr_C, xi_A, xi_B, xi_C)

        # ----------------Fusion Conv Layer--------------------------#
        Xr, Xi = self.Conv_Fu(xr_F3, xi_F3)  # 1 X 1
        Xr, Xi = self.BN_Fu(Xr, Xi)
        Xr, Xi = self.ReLU_Fu(Xr, Xi)#Tensor:(200,128,1,1)

        X = torch.sqrt(torch.pow(Xr, 2) + torch.pow(Xi, 2))

        return X

class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net, self).__init__()
        self.branch1=nn.Sequential(
          ResNet50(num_classes)
     
        )
        self.branch2=nn.Sequential(
            MSCVNet(num_classes)
        )
        self.fc=nn.Linear(256,num_classes,bias=True)

    def forward(self,x,yr):
        out1=self.branch1(x)#out:tensor:(200,32,32,32)
        # print(out1.size())
        out2=self.branch2(yr)
        output=torch.cat((out1,out2),1)

        output=output.view(output.size(0),-1)
        output=self.fc(output)
        return output
#---------------------MyDataset-----------------------
class MyDataset(Dataset):
    def __init__(self, img_r, img_i, label, transform=None):
        super(MyDataset,self).__init__()
        self.img_r = torch.from_numpy(img_r).float()
        self.img_i = torch.from_numpy(img_i).float()
        self.label = torch.from_numpy(label).long()
        self.transform = transform

    def __getitem__(self, index):
        img_r = self.img_r[index]
        img_i = self.img_i[index]
        label = self.label[index]
        return img_r, img_i, label

    def __len__(self):
        return self.img_r.shape[0]

