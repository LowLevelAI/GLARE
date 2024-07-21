import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get
from utils import util
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torch.cuda.amp import GradScaler, autocast
import numpy as np
logger = logging.getLogger('base')

from models.modules.losses import l1_loss, PerceptualNetwork
from models.modules.pytorch_msssim import msssim

class VQLLFLOWDModel(BaseModel):
    def __init__(self, opt, step):
        super(VQLLFLOWDModel, self).__init__(opt)
        self.opt = opt
        
        self.already_print_params_num = False

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt['datasets']['train']['GT_size']  # opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        self.net_hq = networks.find_vqgan(opt).to(self.device)

        vqgan_path= opt['path']['pretrained_vqgan']
        if vqgan_path is not None:
            sd = torch.load(vqgan_path)
            try:
                self.net_hq.load_state_dict(sd, strict=False)
                print(f"net_hq is restored from {vqgan_path}")
                logger.info(f"net_hq is restored from {vqgan_path}")
            except:
                print('loading net_hq error')
        for param in self.net_hq.parameters():
            param.requires_grad = False
        self.net_hq.eval()

        pretrained_netG_path= opt['path']['pretrain_model_G']
        sd = torch.load(pretrained_netG_path)
        #["state_dict"]
        try:
            self.netG.load_state_dict(sd, strict=False)
            print(f"NetG is restored from {pretrained_netG_path}")
            logger.info(f" NetG is restored from {pretrained_netG_path}")
        except:
            print('error')

        
        
        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0:
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
                self.net_hq = DistributedDataParallel(self.net_hq, device_ids=[torch.cuda.current_device()])
                print('DistributedDataParallel')
            elif len(opt['gpu_ids']) > 1:
                self.netG = DataParallel(self.netG, opt['gpu_ids'])
                self.net_hq = DataParallel(self.net_hq, opt['gpu_ids'])
                print('DataParallel')
            else:
                self.netG.to(self.device)
                self.net_hq.to(self.device)


        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()


        self.l1_loss=l1_loss
        self.perceptual_loss=PerceptualNetwork()
        self.msssim = msssim

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        if isinstance(wd_G, str): wd_G = eval(wd_G)
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                else:
                    optim_params_other.append(v)
                    

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': 1e-5}
            ]
        )
        self.optimizers.append(self.optimizer_G)

        

        self.scaler = GradScaler()
        
        
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
    
    def feed_testdata(self, data):
        self.var_L = data.to(self.device)  # LQ
        

    def get_module(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def optimize_color_encoder(self, step):
        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()
        color_lr, color_gt = self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, gt=self.real_H,
                                                                     get_color_map=True)
        losses = {}
        total_loss = (color_gt - color_lr).abs().mean()
        # try:
        total_loss.backward()
        self.optimizer_G.step()
        mean = total_loss.item()
        return mean

    def optimize_parameters(self, step, val_loader):
        torch.cuda.empty_cache()
        

        self.netG.train()
        self.log_dict = OrderedDict()

        self.optimizer_G.zero_grad()

      
        if self.opt['optimize_all_z']:
            if self.opt['gpu_ids'] is not None and len(self.opt['gpu_ids']) > 0:
                epses = [[] for _ in range(len(self.opt['gpu_ids']))]
            else:
                epses = []
        else:
            epses = None
        
        rec, _ = self.netG(net_vq = self.get_module(self.net_hq), lr=self.var_L, reverse=True, reverse_with_grad=True, epses=epses,
                                         align_condition_feature=opt_get(self.opt,
                                                                         ['align_condition_feature']) or False,                                            
                lr_enc=None)   
        rec = rec.to(torch.float32)

        losses={}

        sr = rec.clamp(0, 1)
        not_nan_mask = ~torch.isnan(sr)
        sr[torch.isnan(sr)] = 0

        l1_loss = ((sr- self.real_H) * not_nan_mask).abs().mean()
        losses['l1_loss'] = l1_loss

        losses['percep_loss']  = self.perceptual_loss(sr, self.real_H)*0.01
        losses['ssim_loss'] = (1-self.msssim(sr, self.real_H, normalize=True)) * 0.2
        
        total_loss = sum(losses.values())
        
       
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        
        mean = total_loss.item()
        
        return mean, losses['percep_loss'].item(),  losses['l1_loss'].item(), losses['ssim_loss'].item()
        


    def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)
            

    def print_rrdb_state(self):
        for name, param in self.get_module(self.netG).named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def get_color_map(self):
        self.netG.eval()
        z = self.get_z(0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
        with torch.no_grad():
            color_lr, color_gt = self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, gt=self.real_H,
                                                                         get_color_map=True)
        self.netG.train()
        return color_lr, color_gt

    def test(self):
        self.netG.eval()
        self.fake_H = {}
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                    with torch.no_grad():
                        self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True)
        else:
            z = self.get_z(0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            with torch.no_grad():
                # torch.cuda.reset_peak_memory_stats()
                self.fake_H[(0, 0)], logdet = self.netG(net_vq = self.get_module(self.net_hq), lr=self.var_L, z=z.to(self.var_L.device), eps_std=0, reverse=True)    
        self.netG.train()
        return None


    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        if heat is None:
            heat = 0
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(net_vq = self.net_hq,lr=lq, z=z, eps_std=heat, reverse=True, reverse_with_grad=False, epses=epses)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.get_module(self.netG).flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** L
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            size = (batch_size, 3 * fac * fac, H, W)
            z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float()
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    out_dict[('NORMAL', heat, i)] = self.fake_H[(heat, i)].detach()[0].float()
        else:
            out_dict['NORMAL'] = self.fake_H[(0, 0)].detach()[0].float()
            
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


