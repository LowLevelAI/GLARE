#  [ECCV 2024] GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval [[Paper]](https://arxiv.org/pdf/2407.12431)

<h4 align="center">Han Zhou<sup>1,*</sup>, Wei Dong<sup>1,*</sup>, Xiaohong Liu<sup>2,&dagger;</sup>, Shuaicheng Liu<sup>3</sup>, Xiongkuo Min<sup>2</sup>, Guangtao Zhai<sup>2</sup>, Jun Chen<sup>1,&dagger;</sup></center>
<h4 align="center"><sup>1</sup>McMaster University, <sup>2</sup>Shanghai Jiao Tong University, 
<h4 align="center"><sup>3</sup>University of Electronic Science and Technology of China</center></center>
<h4 align="center"><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Authors</center></center>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glare-low-light-image-enhancement-via/low-light-image-enhancement-on-lol-v2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2?p=glare-low-light-image-enhancement-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glare-low-light-image-enhancement-via/low-light-image-enhancement-on-lol-v2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2-1?p=glare-low-light-image-enhancement-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glare-low-light-image-enhancement-via/low-light-image-enhancement-on-sdsd-indoor)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sdsd-indoor?p=glare-low-light-image-enhancement-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glare-low-light-image-enhancement-via/low-light-image-enhancement-on-sdsd-outdoor)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sdsd-outdoor?p=glare-low-light-image-enhancement-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glare-low-light-image-enhancement-via/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=glare-low-light-image-enhancement-via)




### Introduction
This repository represents the official implementation of our ECCV 2024 paper titled **GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval**. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We present GLARE, a novel network for low-light image enhancement.

- **Codebook-based LLIE**: exploit normal-light (NL) images to extract NL codebook prior as the guidance.
- **Generative Feature Learning**: develop an invertible latent normalizing flow strategy for feature alignment.
- **Adaptive Feature Transformation**: adaptively introduces input information into the decoding process and allows flexible adjustments for users. 
- **Future:** network structure can be meticulously optimized to improve efficiency and performance in the future.

### Overall Framework
![teaser](images/framework.png)

## 📢 News
**2024-09-21:** Inference code for unpaired images and pre-trained models for LOL-v2-real is released! :rocket:<br>
**2024-07-21:** Inference code and pre-trained models for LOL is released! Feel free to use them. ⭐ <br>
**2024-07-21:** [License](LICENSE.txt) is updated to Apache License, Version 2.0. 💫 <br>
**2024-07-19:** Paper is available at: <a href="https://arxiv.org/pdf/2407.12431"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a>. :tada: <br>
**2024-07-01:** Our paper has been accepted by ECCV 2024. Code and Models will be released. :rocket:<br>


## ∞ TODO
- 🔜 Training code.

## 🛠️ Setup

The inference code was tested on:

- Ubuntu 22.04 LTS, Python 3.8, CUDA 11.3, GeForce RTX 2080Ti or higher GPU RAM.

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/LowLevelAI/GLARE.git
cd GLARE
```

### 💻 Dependencies

- **Make Conda Environment: Using [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create the environment:** 

    ```bash
    conda create -n glare python=3.8
    conda activate glare
    ```
- **Then install dependencies:**

  ```bash
  conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
  pip install addict future lmdb numpy opencv-python Pillow pyyaml requests scikit-image scipy tqdm yapf einops tb-nightly natsort
  pip install pyiqa==0.1.4 
  pip install pytorch_lightning==1.6.0
  pip install --force-reinstall charset-normalizer==3.1.0
  ```

- **Build CUDA extensions:**
  
  ```bash
  cd GLARE/defor_cuda_ext
  BASICSR_EXT=True python setup.py develop
  ```

- **Remove CUDA extensions** (/GLARE/defor_cuda_ext/basicsr/ops/dcn/deform_conv_ext.xxxxxx.so) to the path: **/GLARE/code/models/modules/ops/dcn/**.


## 🏃 Testing on benchmark datasets

### 📷 Download following datasets:

LOL [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)



### ⬇ Download pre-trained models

Download [pre-trained weights for LOL](https://drive.google.com/drive/folders/1DuATvqpNgRGlPq5_LvvzghkFdFL9sYvq), [pre-trained weights for LOL-v2-real](https://drive.google.com/drive/folders/1Cesn3jJAdxjT7DDZCTMU8Vt2CnauBL7F?usp=drive_link) and place them to folder `pretrained_weights_lol`, `pretrained_weights_lol-v2-real`, respectively.

### 🚀 Run inference

For LOL dataset

```bash
python code/infer_dataset_lol.py
```

For LOL-v2-real dataset

```bash
python code/infer_dataset_lolv2-real.py
```

For unpaired testing, please make sure the *dataroot_unpaired* in the .yml file is correct.

```bash
python code/infer_unpaired.py
```

You can find all results in `results/`. **Enjoy**!


## 🏋️ Training

Comming Soon~


## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🎓 Citation

Please cite our paper:

```bibtex
@InProceedings{Han_ECCV24_GLARE,
    author    = {Zhou, Han and Dong, Wei and Liu, Xiaohong and Liu, Shuaicheng and Min, Xiongkuo and Zhai, Guangtao and Chen, Jun},
    title     = {GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2024}
}

@article{GLARE,
      title = {GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval}, 
      author = {Zhou, Han and Dong, Wei and Liu, Xiaohong and Liu, Shuaicheng and Min, Xiongkuo and Zhai, Guangtao and Chen, Jun},
      journal = {arXiv preprint arXiv:2407.12431},
      year = {2024}
}
```

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


