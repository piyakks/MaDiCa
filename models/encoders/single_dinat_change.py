import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.encoders.vmamba import Backbone_VSSM
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from models.encoders.dinat_mixer import SaliencyDinatBlock
from models.HA import HA
from models.encoders.dinat.dinat import dinat_small


class RGBXMamba(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=128,
                 pretrained=None,
                 mlp_ratio=0.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[352, 352],
                 patch_size=4,
                 drop_path_rate=0.6,
                 
                 **kwargs):
        super().__init__()
        self.hook_feats = {}
        self.ape = ape
        self.vssm_r = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        # self.gft=GlobalFilter(3,img_size[0],img_size[1]//2+1)
        self.pred_saliency = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, bias=False)
        
        self.saliency_mamba = nn.ModuleList(
            SaliencyDinatBlock(
                dim=dims * (2 ** i),
                mlp_ratio=2.0,
                kernel_size=7,
                dilation=1,                
            ) for i in range(3)
        )

        self.ha = HA()
        
        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)
    def get_hook(self,name):
        def hook_fn(module, input, output):
            self.hook_feats[name] = output
        return hook_fn
    def forward_features(self, x_rgb):
        """
        x_rgb: B x C x H x W
        """
        outs_fused=[]
        B = x_rgb.shape[0]
        
        # fft=self.gft(x_rgb)
        # x = self.backbone(x_rgb) # B x C x H x W
        # outs_rgb = self.hook_feats
        outs_rgb = self.vssm_r(x_rgb)
            
            # outs_rgb[i]=outs_rgb[i]+fft
        saliency = self.pred_saliency(F.interpolate(outs_rgb[3], scale_factor=8, mode='bilinear', align_corners=False))
        # saliency = self.pred_saliency(F.interpolate(outs_rgb[3].permute(0,3,1,2), scale_factor=8, mode='bilinear', align_corners=False))
        guide_saliency = torch.nn.Sigmoid()(saliency)
        guide_saliency = self.ha(guide_saliency)
        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
            else:
                out_rgb = outs_rgb[i].permute(0, 2, 3, 1).contiguous()
            if i < 3:
                B,C,H,W = out_rgb.shape
                resized_gt = F.interpolate(guide_saliency, size=(H, W), mode='bilinear', align_corners=False)
                resized_gt  = (resized_gt >= 0.3).float()

                out_rgb = self.saliency_mamba[i](out_rgb, resized_gt)

            out_rgb = out_rgb.permute(0, 3, 1, 2).contiguous()
            # print(out_rgb.shape)
            outs_fused.append(out_rgb)
        
        
        
                
        return outs_fused, saliency

    def forward(self, x_rgb):
        out, saliency = self.forward_features(x_rgb)
        return out, saliency

class vssm_tiny(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='models/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            # depths=[3, 4, 18, 5],
            depths=[2,2,27,2],
            dims=96,
            pretrained='models/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXMamba):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='models/pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
