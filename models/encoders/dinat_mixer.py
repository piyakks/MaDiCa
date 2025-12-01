import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from natten import NeighborhoodAttention2D as NeighborhoodAttention

from natten.context import is_fna_enabled
from natten.experimental import na2d as experimental_na2d
from natten.functional import na2d, na2d_av, na2d_qk
from natten.types import CausalArg2DTypeOrDed, Dimension2DTypeOrDed
from natten.utils import check_all_args, log

    
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.reduction = reduction
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride, groups=out_channels)

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
class SaliencyDinatBlock(nn.Module):
    '''
    Saliency Mamba (SaMB), with 2d SSM
    '''
    def __init__(
        self,
        dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        kernel_size: int = 11,
        dilation: int = 2,
        num_heads: int= 4,
        qkv_bias=False,
        qk_scale=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # self.norm = norm_layer(hidden_dim)
        
        self.op=NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop,
            rel_pos_bias=True,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)
        self.norm1=norm_layer(dim)
    def _forward(self, x: torch.Tensor, gt: torch.Tensor):
        b,w,h,c=x.shape
        
        gt = F.interpolate(gt.float(), size=(w, h), mode='nearest') 
        gt = gt.bool()
        saliency_scale=torch.where(gt, torch.tensor(1.2, device=gt.device), torch.tensor(1.0, device=gt.device)).permute(0,2,3,1)
        
        scale_x=x*saliency_scale
       
        forward = self.drop_path(self.op(scale_x))
        
        # back= self.drop_path(self.op2(x, back_gt))
        # y= self.drop_path(self.op2(x, back_gt))
        x=x+forward
        
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN

        return x

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, gt)
        else:
            return self._forward(x, gt)
        
        
# class NeighborhoodAttention(nn.Module):
#     """
#     Neighborhood Attention 2D Module
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         kernel_size: Dimension2DTypeOrDed,
#         dilation: Dimension2DTypeOrDed = 1,
#         is_causal: CausalArg2DTypeOrDed = False,
#         rel_pos_bias: bool = False,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         use_experimental_ops: bool = False,
#     ):
#         super().__init__()
#         kernel_size_, dilation_, is_causal_ = check_all_args(
#             2, kernel_size, dilation, is_causal
#         )
#         assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 2
#         if any(is_causal_) and rel_pos_bias:
#             raise NotImplementedError(
#                 "Causal neighborhood attention is undefined with positional biases."
#                 "Please consider disabling positional biases, or open an issue."
#             )

#         self.num_heads = num_heads
#         self.head_dim = dim // self.num_heads
#         self.scale = qk_scale or self.head_dim**-0.5
#         self.kernel_size = kernel_size_
#         self.dilation = dilation_
#         self.is_causal = is_causal_

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         if rel_pos_bias:
#             self.rpb = nn.Parameter(
#                 torch.zeros(
#                     num_heads,
#                     (2 * self.kernel_size[0] - 1),
#                     (2 * self.kernel_size[1] - 1),
#                 )
#             )
#             trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
#             self.rpb_mask = nn.Parameter(
#                 torch.zeros(
#                     1,
#                     (2 * self.kernel_size[0] - 1),
#                     (2 * self.kernel_size[1] - 1),
#                 )
#             )
#             # trunc_normal_(self.rpb_mask, std=0.02, mean=0.0, a=-2.0, b=2.0)


#         else:
#             self.register_parameter("rpb", None)
#         self.attn_drop_rate = attn_drop
#         self.attn_drop = nn.Dropout(self.attn_drop_rate)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.use_experimental_ops = use_experimental_ops

#     def forward(self, x: Tensor,mask:Tensor) -> Tensor: 
#         if x.dim() != 4:
#             raise ValueError(
#                 f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
#             )

#         B, H, W, C = x.shape

#         if is_fna_enabled():
#             if self.attn_drop_rate > 0:
#                 logger.error(
#                     "You're using fused neighborhood attention, and passed in a "
#                     "non-zero attention dropout rate. This implementation does "
#                     "support attention dropout yet, which means dropout is NOT being applied "
#                     "to your attention weights."
#                 )

#             qkv = (
#                 self.qkv(x)
#                 .reshape(B, H, W, 3, self.num_heads, self.head_dim)
#                 .permute(3, 0, 1, 2, 4, 5)
#             )
#             q, k, v = qkv[0], qkv[1], qkv[2]
#             x = (experimental_na2d if self.use_experimental_ops else na2d)(
#                 q,
#                 k,
#                 v,
#                 kernel_size=self.kernel_size,
#                 dilation=self.dilation,
#                 is_causal=self.is_causal,
#                 rpb=self.rpb,
#                 scale=self.scale,
#             )
#             x = x.reshape(B, H, W, C)

#         else:
#             if self.use_experimental_ops:
#                 raise NotImplementedError(
#                     "Only fused NA is included in experimental support for torch.compile and torch's FLOP counter."
#                 )
#             one = torch.ones_like(mask)
#             xm = torch.where(mask < 0.1, one, one*2)
#             xm=xm.unsqueeze(1)
#             mm = na2d_qk(
#                 xm,
#                 xm,
#                 kernel_size=self.kernel_size,
#                 dilation=self.dilation,
#                 is_causal=self.is_causal,
#                 rpb=self.rpb_mask,
#             ) 
#             one = torch.ones_like(mm)
#             mm = torch.where(mm==1, one*0.2, one)


#             qkv = (
#                 self.qkv(x)
#                 .reshape(B, H, W, 3, self.num_heads, self.head_dim)
#                 .permute(3, 0, 4, 1, 2, 5)
#             )
#             q, k, v = qkv[0], qkv[1], qkv[2]
#             q = q * self.scale
#             attn = na2d_qk(
#                 q,
#                 k,
#                 kernel_size=self.kernel_size,
#                 dilation=self.dilation,
#                 is_causal=self.is_causal,
#                 rpb=self.rpb,
#             )
#             attn=attn*mm   
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = na2d_av(
#                 attn,
#                 v,
#                 kernel_size=self.kernel_size,
#                 dilation=self.dilation,
#                 is_causal=self.is_causal,
#             )
#             x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

#         return self.proj_drop(self.proj(x))

#     def extra_repr(self) -> str:
#         return (
#             f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
#             + f"kernel_size={self.kernel_size}, "
#             + f"dilation={self.dilation}, "
#             + f"is_causal={self.is_causal}, "
#             + f"has_bias={self.rpb is not None}"
#         )
