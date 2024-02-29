import torch.nn as nn
import torch

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
    
    

import torch
class SpatialNorm(nn.Module):
    def __init__(self, f_channels, zq_channels, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False, **norm_layer_params):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv1d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv1d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv1d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, f, zq):
        # print("f",f.shape)
        f_size = f.shape[-1:]
        # print(f_size)
        # zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        # zq = nn.Upsample(zq, size=f_size, mode='nearest')
        if self.add_conv:
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

def Normalize(in_channels, zq_ch, add_conv):
    return SpatialNorm(in_channels, zq_ch, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=add_conv, num_groups=32, eps=1e-6, affine=True)

class ResConv1DBlock_decoder(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None, zq_ch=None, add_conv=False):
        super().__init__()
        padding = dilation

        
        self.norm1 = Normalize(n_in, zq_ch, add_conv=add_conv)
        self.norm2 = Normalize(n_in, zq_ch, add_conv=add_conv)

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     

    def forward(self, x , zq):
        # print("zq", zq.shape)
        x_orig = x
        x = self.norm1(x, zq)
        x = self.activation1(x)
            
        x = self.conv1(x)

        x = self.norm2(x, zq)
        x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

# class Resnet1D_decoder(nn.Module):
#     def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, zq_ch=None, add_conv=False):
#         super().__init__()
        
#         blocks = [ResConv1DBlock_decoder(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, zq_ch=zq_ch, add_conv=add_conv) for depth in range(n_depth)]
#         if reverse_dilation:
#             blocks = blocks[::-1]
        
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x, zq):        
#         return self.model(x, zq)
class Resnet1D_decoder(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, zq_ch=None, add_conv=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([ResConv1DBlock_decoder(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, zq_ch=zq_ch, add_conv=add_conv) for depth in range(n_depth)])
        self.reverse_dilation = reverse_dilation

    def forward(self, x, zq):        
        if self.reverse_dilation:
            for block in reversed(self.blocks):
                x = block(x, zq)
        else:
            for block in self.blocks:
                x = block(x, zq)
        return x