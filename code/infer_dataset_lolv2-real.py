import glob
import sys
from collections import OrderedDict
from models.modules.VQModel_arch import VQModel
from natsort import natsort
import argparse
import options.options as option
from Measure import Measure, psnr

from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
import pyiqa
from skimage import img_as_ubyte
from utils.utils2 import PSNR, calculate_ssim, save_img
def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get('NORMAL'))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')

def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="./code/confs/LOL-v2-real.yml")
    args = parser.parse_args()
    conf_path = args.opt
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    device = 'cuda'
    model.netG = model.netG.to(device)
    model.net_hq = model.net_hq.to(device)


    lr_dir = opt['dataroot_LR']
    hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results', conf)
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)
    fname = str(conf)+'.csv'
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    else:
        df = None

    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
        hr = imread(hr_path)
        hr_t = t(hr)
        his = hiseq_color_cv2_img(lr)
        if opt.get("histeq_as_input", False):
            lr = his
        
        # Pad image 
        h, w, c = lr.shape
        lr = impad(lr, bottom=int(20), left=int(20))
        
        lr_t = t(lr)
        if opt["datasets"]["train"].get("log_low", False):
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        if opt.get("concat_histeq", False):
            his = t(his)
            lr_t = torch.cat([lr_t, his], dim=1)

        
        with torch.cuda.amp.autocast():
            lr_out = model.get_sr(lq=lr_t.to(device), heat=None)[:, :, :h, 20:]

        restored = torch.clamp(lr_out, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        target = hr / 255
            
        mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        restored = np.clip(restored * (mean_target / mean_restored), 0, 1)
        

        meas = OrderedDict(conf=conf, name=str(os.path.basename(hr_path)))


        meas['PSNR'] = PSNR(target, restored)
        meas['SSIM'] = calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored))
        meas['LPIPS'] = measure.lpips(img_as_ubyte(restored), img_as_ubyte(target))
            
        save_path = os.path.join(test_dir, os.path.basename(hr_path))

        save_img(save_path, img_as_ubyte(restored))

        str_out = format_measurements(meas)
        print(str_out)

        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])
    df.to_csv(path_out_measures_final, index=False)        
    str_out = format_measurements(df.mean(numeric_only=True))
    print(f"Results in: {path_out_measures_final}")
    print('Mean: ' + str_out)
    with open(os.path.join(test_dir, 'metrics_lol-v2-real.txt'),'a') as f:
        f.write(str(conf) + ' ' + str(str_out) + '\n')
      

def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.4f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
