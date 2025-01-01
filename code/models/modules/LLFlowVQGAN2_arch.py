
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.ConditionEncoder import ConEncoder1, NoEncoder
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from models.modules.color_encoder import ColorEncoder
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast

class LLFlowVQGAN2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32, scale=4, latent_size=64, latent_channel=512, K=None, opt=None, step=None):
        super(LLFlowVQGAN2, self).__init__()
        self.crop_size = opt['datasets']['train']['GT_size']
        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])   #255
        
        if opt['cond_encoder'] == 'ConEncoder1':                                                    #self.RRDB = ConEncoder1
            self.RRDB = ConEncoder1(opt=opt)
        else:
            print('WARNING: Cannot find the conditional encoder %s, select RRDBNet by default.' % opt['cond_encoder'])
            # if self.opt['encode_color_map']: print('Warning: ''encode_color_map'' is not implemented in RRDBNet')
            opt['cond_encoder'] = 'RRDBNet'
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)

        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])     #64
        hidden_channels = hidden_channels or 64

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((80, 80, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)       #coupling: CondAffineSeparatedAndCond

    @autocast()
    def forward(self, lr=None, gt=None,  z=None, eps_std=None, reverse=True, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None, align_condition_feature=False, get_color_map=False):
        if get_color_map:
            color_lr = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            return color_lr, color_gt
        if not reverse:
            #gt=self.unsqueeze2d(gt,4)
            #print('gt', gt.shape)
            if epses is not None and gt.device.index is not None:
                epses = epses[gt.device.index]
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label, align_condition_feature=align_condition_feature)
        else:
            # assert lr.shape[0] == 1
            assert lr.shape[1] == 3 or lr.shape[1] == 6
            # assert lr.shape[2] == 20
            # assert lr.shape[3] == 20
            # assert z.shape[0] == 1
            # assert z.shape[1] == 3 * 8 * 8
            # assert z.shape[2] == 20
            # assert z.shape[3] == 20
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None,
                    align_condition_feature=False):
        
        if lr_enc is None and self.RRDB:
            lr_enc = self.RRDB(lr)    #执行
        #print('conditionalencoder',lr_enc['color_map'].shape)  

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            print('noiseQuant:', noiseQuant)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot)

        objective = logdet.clone()
        #print(epses.shape)  #torch.Size([2, 48, 16, 16])
        z = epses
        if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']:
            if 'avg_pool_color_map' in self.opt.keys() and self.opt['avg_pool_color_map']:
                mean = F.avg_pool2d(lr_enc['color_map'], 7, 1, 3)if random.random() > self.opt[
                    'train_gt_ratio'] else F.avg_pool2d(
                    gt / (gt.sum(dim=1, keepdims=True) + 1e-4), 7, 1, 3)
        else:
            if self.RRDB is not None:
                mean = lr_enc['color_map'] if random.random() > self.opt['train_gt_ratio'] else gt
            else:
                mean = lr[:,:3]
        #mean = lr_enc['color_map']

        #print('before', z.shape)
        objective = objective + flow.GaussianDiag.logp(mean, torch.tensor(0.).to(z.device), z)
        #print('after', objective)   #tensor([-104803.0078, -226577.6562], device='cuda:0'
        #print('pixels', pixels)
        nll = (-objective) / float(np.log(2.) * pixels)
    
        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet


    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None and self.RRDB:
            lr_enc = self.RRDB(lr)
        
        z = lr_enc['color_map']
        #print('z', z.shape)   #[4,3 64, 64]
        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)
        #x=self.squeeze2d(x, 4)
        if self.opt['encode_color_map']:
            color_map = self.color_map_encoder(lr)
            color_out = nn.functional.avg_pool2d(x, 11, 1, 5)
            color_out = color_out / torch.sum(color_out, 1, keepdim=True)
            x = x * (color_map / color_out)
        if self.opt['to_yuv']:
            x = self.yuv2rgb(x)
        return x, logdet
