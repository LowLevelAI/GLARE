import math
import lpips
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import vgg16

def l1_loss(pred, target):
    return  F.smooth_l1_loss(pred, target)

class PerceptualNetwork(torch.nn.Module):
    def __init__(self):
        super(PerceptualNetwork, self).__init__()
        self.vgg_model = vgg16(pretrained=True)
        self.vgg_model = self.vgg_model.features[:16].to('cuda')
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return sum(loss)/len(loss)