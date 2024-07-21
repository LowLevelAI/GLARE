import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from models.modules.encoder_decoder import AttnBlock, ResnetBlock, Upsample, nonlinearity, Normalize
from models.modules.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv

class SecondDecoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[64], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=3):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.warp = nn.ModuleList()   
        self.warp.append(Feat_Transform2(256, m=-1))
        self.warp.append(Feat_Transform2(128, m=-0.6))
        self.residual_conv = torch.nn.Conv2d(128,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, code_decoder_output):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level!=2:
                #print(i_level)
                code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to('cuda')

                # code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to(torch.float32)
                # h = h.to(torch.float32)
                #print(h.shape)
                #print(code_decoder_output[1-i_level].shape)
                x_vq, weight = self.warp[1-i_level](code_decoder_output[1-i_level], h)
                h = x_vq * weight + h
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.residual_conv(h)
        return h





class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        out =out.to(torch.float32)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        
        offset = torch.cat((o1, o2), dim=1)
        
        mask = torch.sigmoid(mask)

        
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)
def swish(x):
    return x*torch.sigmoid(x)    
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in
class Feat_Transform(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.instance = nn.InstanceNorm2d(in_channel, affine=True)
        self.instance_conv = nn.Conv2d(in_channel, in_channel, 3, 1,1)

        self.encode_enc = ResBlock(2*in_channel, in_channel)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))
        
        self.weight = nn.Sequential(
                    nn.Conv2d(2*in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))
        
        self.activation = nn.Sigmoid()

    def forward(self, x_vq, x_f):
        out_instance = self.instance(x_vq)
        out_vq = self.instance_conv(out_instance)

        combine_feat = self.encode_enc(torch.cat([x_vq, x_f],1))
        scale = self.scale(combine_feat)
        shift = self.shift(combine_feat)
        #print(scale.mean())
        #print(shift.mean())
        out_vq = out_vq * scale + shift

        weight = self.weight(torch.cat([out_vq, x_f] ,1))
        weight = x_f.mean()/out_vq.mean()
        return out_vq, weight
    
  
class Feat_Transform2(nn.Module):
    def __init__(self, in_channel, m=-0.80):
        super().__init__()
        #self.instance = nn.InstanceNorm2d(in_channel, affine=True)
        #self.instance_conv = nn.Conv2d(in_channel, in_channel, 3, 1,1)

        self.encode_enc = ResBlock(2*in_channel, in_channel)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))
        
        self.weight = nn.Sequential(
                    nn.Conv2d(2*in_channel, in_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1))
        

        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.activation = nn.Sigmoid()


    def ins_norm(self, x):
        mean = torch.mean(x, dim=[2,3])
        mean = mean.unsqueeze(-1).unsqueeze(-1).expand(x.shape)

        std = torch.std(x, dim=[2,3])
        std = std.unsqueeze(-1).unsqueeze(-1).expand(x.shape)

        x = (x-mean)/ std 
        return x
    

    def forward(self, x_vq, x_f):
        ins_vq = self.ins_norm(x_vq)
        
        feat_combine = self.encode_enc(torch.cat([x_vq, x_f], dim=1))

        scale = self.scale(feat_combine)
        shift = self.shift(feat_combine)

        out_vq = ins_vq* scale + shift

        weight = self.activation(self.w)
        return out_vq, weight

       


class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1)
        self.dcn = DCNv2Pack(in_channel, in_channel, 3, padding=1, deformable_groups=4)

    def forward(self, x_vq, x_residual):
        x_residual = self.offset(torch.cat([x_vq, x_residual], dim=1))
        
        feat_after_warp = self.dcn(x_vq, x_residual)

        return feat_after_warp
    
class MultiScaleDecoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[64], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=3):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.warp = nn.ModuleList()   
        self.warp.append(WarpBlock(256))
        self.warp.append(WarpBlock(128))
        self.residual_conv = torch.nn.Conv2d(128,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, code_decoder_output):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level!=2:
                #print(i_level)
                code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to('cuda')

                code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to(torch.float32)
                h = h.to(torch.float32)
                #print(h.shape)
                #print(code_decoder_output[1-i_level].shape)
                x_vq = self.warp[1-i_level](code_decoder_output[1-i_level], h)
                h = h + x_vq *(h.mean() / x_vq.mean())
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.residual_conv(h)
        return h
    



class MultiScaleDecoder2(nn.Module):
    def __init__(self, ch=64, out_ch=3, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[64], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=3):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.warp = nn.ModuleList()   
        self.warp.append(WarpBlock(ch*2))
        self.warp.append(WarpBlock(ch))
        self.residual_conv = torch.nn.Conv2d(ch,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        scale_1 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())
        
        scale_2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.Sigmoid())
        bias_1 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())
        bias_2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.Sigmoid())
        
        self.scale = nn.ModuleList()
        self.scale.append(scale_1)
        self.scale.append(scale_2) 

        self.bias = nn.ModuleList()
        self.bias.append(bias_1)
        self.bias.append(bias_2) 

        self.enc = nn.ModuleList()
        self.enc.append(ResBlock(512, 256))
        self.enc.append(ResBlock(256,128))

        self.mix = nn.ModuleList()

        self.mix.append(Mix(m=-1.0))
        self.mix.append(Mix(m=-0.6))
         

    def forward(self, z, code_decoder_output, enc_feat):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level!=2:
                #print(i_level)
                code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to('cuda')

                code_decoder_output[1-i_level]=code_decoder_output[1-i_level].to(torch.float32)
                h = h.to(torch.float32)
                #print(code_decoder_output[1-i_level].shape)
                #print(enc_feat[i_level].shape)
                # if i_level==1:
                    
               
                #combine_feat = self.enc[1-i_level](torch.cat([code_decoder_output[1-i_level], enc_feat[i_level]], 1))
                #scale = self.scale[1-i_level](enc_feat[i_level]).mean()
                #bias = self.bias[1-i_level](enc_feat[i_level]).mean()
                #print(scale.mean())
                #print(bias.mean())
                
                h = self.mix[1-i_level](enc_feat[i_level], h)

                x_vq = self.warp[1-i_level](code_decoder_output[1-i_level], h)
                #ratio = torch.mean(h, dim=[2,3]).unsqueeze(-1).unsqueeze(-1).expand(h.shape)/torch.mean(x_vq, dim=[2,3]).unsqueeze(-1).unsqueeze(-1).expand(x_vq.shape)
                h = h + x_vq * (h.mean() / x_vq.mean())
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.residual_conv(h)
        return h
    

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
    

 

    