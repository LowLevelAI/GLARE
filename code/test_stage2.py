import glob
import sys
from collections import OrderedDict
from models.modules.VQModel_arch import VQModel
from natsort import natsort
import argparse
import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2


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
    parser.add_argument("--opt", default="./code/confs/LOL-vq-no-squeeze-with-mean-320-large-lr-gt-hqps-9-17-4.yml")
    args = parser.parse_args()
    conf_path = args.opt
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    device = 'cuda:1'
    model.netG = model.netG.to(device)
    model.net_hq=model.net_hq.to(device)


    lr_dir = opt['dataroot_LR']
    hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results-LOL-2024-1203', conf)
    print(f"Out dir: {test_dir}")


    scale = opt['scale']

    pad_factor = 2
    heat = 0

   
    for i in natsort.natsorted(os.listdir('experiments/train_stage2_LOL/models/')):
        path=os.path.join('experiments/train_stage2_LOL/models/', i)
        name=os.path.splitext(i)[0]
        print('testing on '+ str(name))
        model.load_network(load_path=path, network=model.netG)
        model.netG = model.netG.to(device)
        model.net_hq=model.net_hq.to(device)
        

        measure = Measure(use_gpu=False)

        fname = str(i)+'.csv'
        fname_tmp = fname + "_"
        path_out_measures = os.path.join(test_dir, fname_tmp)
        path_out_measures_final = os.path.join(test_dir, fname)

        if os.path.isfile(path_out_measures_final):
            df = pd.read_csv(path_out_measures_final)
        elif os.path.isfile(path_out_measures):
            df = pd.read_csv(path_out_measures)
        else:
            df = None

        for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

            lr = imread(lr_path)
            hr = imread(hr_path)
            his = hiseq_color_cv2_img(lr)
            if opt.get("histeq_as_input", False):
                lr = his
        
            # Pad image to be % 2
            h, w, c = lr.shape
            lq_orig = lr.copy()
            lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                    right=int(np.ceil(w / pad_factor) * pad_factor - w))
        
            lr_t = t(lr)
            if opt["datasets"]["train"].get("log_low", False):
                lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
            if opt.get("concat_histeq", False):
                his = t(his)
                lr_t = torch.cat([lr_t, his], dim=1)

        
            lr_left = lr_t
        
            with torch.cuda.amp.autocast():
                sr_left = model.get_sr(lq=lr_left.to(device), heat=None)
                lr_out = sr_left.to(device)
                with torch.no_grad():
                    lr_out, _, _ = model.net_hq.decode(lr_out)

            path_out_sr = os.path.join(test_dir,  str(name),'no-adjust', os.path.basename(hr_path))

            imwrite(path_out_sr, rgb(lr_out))   
        
            
            #print(lr_out.shape)
            mean_out = lr_out.reshape(lr_out.shape[0],-1).mean(dim=1)
            mean_gt = cv2.cvtColor(hr.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()/255
            out = rgb(torch.clamp(lr_out*(mean_gt/mean_out), 0, 1))
            final_path_adjust = os.path.join(test_dir, str(name), 'adjust', os.path.basename(hr_path))
            imwrite(final_path_adjust, out)

            meas = OrderedDict(conf=conf, heat=heat, name=str(os.path.basename(hr_path)))
            
            meas['Adjust_PSNR'], _, meas['Adjust_LPIPS'] = measure.measure(out, hr)
            meas['Adjust_SSIM'] = measure.ssim(out, hr, gray_scale=False)
            meas['PSNR'], _, meas['LPIPS'] = measure.measure(rgb(lr_out), hr)
            meas['SSIM'] = measure.ssim(rgb(lr_out), hr, gray_scale=False)
            
            

            str_out = format_measurements(meas)
            print(str_out)

            df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])
        df.to_csv(path_out_measures, index=False)
        #os.rename(path_out_measures, path_out_measures_final)
        str_out = format_measurements(df.mean(numeric_only=True))
        print(f"Results in: {path_out_measures_final}")
        print('Mean: ' + str_out)
        with open('results-LOL-2024-1203.txt','a') as f:
            f.write(str(i)+str(str_out)+'\n')


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.4f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
