
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.ConditionEncoder import ConEncoder1
from models.modules.deformableDecoder_arch import MultiScaleDecoder, SecondDecoder, MultiScaleDecoder2
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast

class VQLLFLOWDeformable(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32, scale=4, latent_size=64, latent_channel=512, K=None, opt=None, step=None, fix_modules=['RRDB','flowUpsamplerNet']):
        super(VQLLFLOWDeformable, self).__init__()
        self.crop_size = opt['datasets']['train']['GT_size']
        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])   #255
                                                            #self.RRDB = ConEncoder1
        self.RRDB = ConEncoder1(opt=opt)
        
        
        self.deformable_decoder = MultiScaleDecoder2(ch=128, out_ch=3, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[64], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=3
                 )
        

        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])     #64
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])    #0.5
        set_RRDB_to_train = False
        if set_RRDB_to_train and self.RRDB:
            self.set_rrdb_training(True)

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((80, 80, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)       #coupling: CondAffineSeparatedAndCond
        self.i = 0
        
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def rgb2yuv(self, rgb):
        rgb_ = rgb.transpose(1, 3)  # input is 3*n*n   default
        yuv = torch.tensordot(rgb_, self.A_rgb2yuv, 1).transpose(1, 3)
        return yuv

    def yuv2rgb(self, yuv):
        yuv_ = yuv.transpose(1, 3)  # input is 3*n*n   default
        rgb = torch.tensordot(yuv_, self.A_yuv2rgb, 1).transpose(1, 3)
        return rgb

    def squeeze2d(self, input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x


    def unsqueeze2d(self, input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

    @autocast()
    def forward(self, net_vq=None, gt=None, lr=None, z=None, eps_std=None, reverse=True, epses=None, reverse_with_grad=True,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None, align_condition_feature=False, get_color_map=False):
        if get_color_map:
            color_lr = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            return color_lr, color_gt
        if not reverse:
            
            if epses is not None and gt.device.index is not None:
                epses = epses[gt.device.index]
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label, align_condition_feature=align_condition_feature)
        else:
            
            assert lr.shape[1] == 3 or lr.shape[1] == 6
            
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise, net_vq=net_vq)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise, net_vq=net_vq)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None,
                    align_condition_feature=False, net_vq=None):
        if self.opt['to_yuv']:
            gt = self.rgb2yuv(gt)
        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)    
        print('conditionalencoder',lr_enc['fea_up-1'].shape)  
        print('conditionalencoder',lr_enc['fea_up0'].shape)  
        print('conditionalencoder',lr_enc['color_map'].shape)   
        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot)

        objective = logdet.clone()
        
        z = epses
        if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']:
            if 'avg_pool_color_map' in self.opt.keys() and self.opt['avg_pool_color_map']:
                mean = squeeze2d(F.avg_pool2d(lr_enc['color_map'], 7, 1, 3), 4) if random.random() > self.opt[
                    'train_gt_ratio'] else squeeze2d(F.avg_pool2d(
                    gt / (gt.sum(dim=1, keepdims=True) + 1e-4), 7, 1, 3), 4)
        else:
            if self.RRDB is not None:
                
                mean = squeeze2d(lr_enc['color_map'], 4) if random.random() > self.opt['train_gt_ratio'] else squeeze2d(
                gt/(gt.sum(dim=1, keepdims=True) + 1e-4), 4)
                
            else:
                mean = squeeze2d(lr[:,:3],8)
        
        objective = objective + flow.GaussianDiag.logp(mean, torch.tensor(0.).to(z.device), z)
        
        nll = (-objective) / float(np.log(2.) * pixels)
        if self.opt['encode_color_map']:                #false
            color_map = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            color_loss = (color_gt - color_map).abs().mean()
            nll = nll + color_loss
        if align_condition_feature:      #false
            with torch.no_grad():
                gt_enc = self.rrdbPreprocessing(gt)
            for k, v in gt_enc.items():
                if k in ['fea_up-1']:  # ['fea_up2','fea_up1','fea_up0','fea_up-1']:
                    if self.opt['align_maxpool']:
                        nll = nll + (self.max_pool(gt_enc[k]) - self.max_pool(lr_enc[k])).abs().mean() * (
                            self.opt['align_weight'] if self.opt['align_weight'] is not None else 1)
                    else:
                        nll = nll + (gt_enc[k] - lr_enc[k]).abs().mean() * (
                            self.opt['align_weight'] if self.opt['align_weight'] is not None else 1)
        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)     
        
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            low_level_features = [rrdbResults["block_{}".format(idx)] for idx in block_idxs]
            
            concat = torch.cat(low_level_features, dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True, net_vq=None):

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None and self.RRDB:
            with torch.no_grad():
                lr_enc = self.RRDB(lr, mid_feat =True)
        if self.opt['cond_encoder'] == "NoEncoder":
            z = lr[:,:3]
        else:
            if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']:
                z = F.avg_pool2d(lr_enc['color_map'], 7, 1, 3)
            else:
                z = lr_enc['color_map']
        with torch.no_grad():
            x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)
            enc_feat=x
            x= x.to('cuda:0')

            rec, _, code_decoder_output = net_vq.decode(x)
            
            c_feat = lr_enc['mid_feat']
        rec_deformable = self.deformable_decoder(enc_feat, code_decoder_output, c_feat)      
        return rec_deformable, enc_feat
