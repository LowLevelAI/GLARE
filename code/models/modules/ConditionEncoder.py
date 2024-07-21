

from torchvision.utils import save_image
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from utils.util import opt_get
from models.modules.flow import Conv2dZeros
from models.modules.encoder_decoder import Encoder, Decoder


class ConEncoder1(nn.Module):
    def __init__(self, 
                 resolution=256,
                 ckpt_path='pretrained_weights/vqgan.pkl',
                 double_z=False,
                 z_channels=3,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],
                 num_res_blocks=2,
                 attn_resolutions=[64],
                 dropout=0.0,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw, 
                opt=None):
        self.opt = opt
        
        super(ConEncoder1, self).__init__()

        
        self.encoder = Encoder(ch, out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=True, 
                 in_channels=in_channels, resolution=resolution, z_channels= z_channels, double_z=double_z)
        self.color_conv = nn.Conv2d(3, 3, 3, 1, 1)

        self.cond_conv = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.Sigmoid())     #9.20 night    no sigmoid,     add sigmoid when using 9-17-4-hqps
        

    def forward(self, x, mid_feat = False):
        enc_feat, mid_feat = self.encoder(x, mid_feat=True)
        cond_feat = self.cond_conv(enc_feat)  
        color_feat = self.color_conv(enc_feat)
        results={'cond_feat': cond_feat,
                 'color_map': color_feat
                 }
        if mid_feat:
            results["mid_feat"] = mid_feat
        return results


