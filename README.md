#  [ECCV 2024] GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval [[Paper]](https://arxiv.org/pdf/2407.12431)

<h4 align="center">Han Zhou<sup>1,*</sup>, Wei Dong<sup>1,*</sup>, Xiaohong Liu<sup>2,&dagger;</sup>, Shuaicheng Liu<sup>3</sup>, Xiongkuo Min<sup>2</sup>, Guangtao Zhai<sup>2</sup>, Jun Chen<sup>1,&dagger;</sup></center>
<h4 align="center"><sup>1</sup>McMaster University, <sup>2</sup>Shanghai Jiao Tong University, 
<h4 align="center"><sup>3</sup>University of Electronic Science and Technology of China,</center></center>
<h4 align="center"><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Authors</center></center>


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
2024-07-21: Inference code is released!
2024-07-21: Updated [license](LICENSE.txt) to Apache License, Version 2.0.<br>
2024-07-19: Added arXiv version paper: <a href="https://arxiv.org/pdf/2407.12431"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a>. <br>
2024-07-01: Accepted to ECCV 2024.<br>




