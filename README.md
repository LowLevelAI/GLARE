#  [ECCV 2024] GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval [[Paper]](https://arxiv.org/pdf/2407.12431)

<h4 align="center">Han Zhou<sup>1,*</sup>, Wei Dong<sup>1,*</sup>, Xiaohong Liu<sup>2,&dagger</sup>, Shuaicheng Liu<sup>3</sup></center>
<h4 align="center">1.Sichuan University, 2.Southwest Jiaotong University, 
<h4 align="center">3.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">4.Shanghai Jiaotong University, 5.Megvii Technology</center></center>

### Introduction
This repository represents the official implementation of the paper titled **GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval**. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We present GLARE, a novel network for low-light image enhancement.

- **Codebook-based LLIE**: exploit normal-light (NL) images to extract NL codebook prior as the guidance.
- **Generative Feature Learning**: develop an invertible latent normalizing flow strategy for feature alignment.
- **Adaptive Feature Transformation**: adaptively introduces input information into the decoding process and allows flexible adjustments for users. 
- **Future:** network structure can be meticulously optimized to improve efficiency and performance in the future.

### Overall Framework
![teaser](images/GLARE.png)


