ControlNet Distillation Research in PyTorch
========

**This repository is based on the [ExplainabilityAI ControlNet-PyTorch](https://github.com/explainingai-code/ControlNet-PyTorch) repository and extends it with advanced distillation techniques for fast single-step ControlNet inference.**

## Purpose

This research repository implements and compares different distillation approaches for ControlNet models, enabling **single-step sampling** instead of the traditional multi-step denoising process. The main contributions are:

1. **Consistency ControlNet Distillation**: Single-step sampling using consistency loss
2. **Distribution Matching ControlNet Distillation**: Single-step sampling using distribution matching loss  
3. **Comprehensive Evaluation**: Side-by-side comparison of different distillation approaches

The goal is to achieve **fast inference** (1 step vs 1000 steps) while maintaining high-quality controllable image generation.

## ControlNet Tutorial Video
<a href="https://www.youtube.com/watch?v=n6CwImm_WDI">
   <img alt="ControlNet Tutorial" src="https://github.com/user-attachments/assets/00bcedd4-45b9-4c4f-8563-8c00589e6a08"
   width="400">
</a>

## Sample Output for ControlNet with DDPM on MNIST and with LDM on CelebHQ
Canny Edge Control - Top, Sample - Below

<img src="https://github.com/user-attachments/assets/a52c53b3-b62e-4535-affa-1ee0d0321223" width="500">

___

## Research Focus: ControlNet Distillation

This repository provides comprehensive implementations for:
* **Baseline Models**: Training and Inference of DDPM ControlNet on MNIST and LDM ControlNet on CelebHQ
* **Consistency ControlNet**: Single-step sampling using consistency distillation from pre-trained ControlNet DDPM
* **Distribution Matching ControlNet**: Single-step sampling using distribution matching distillation from pre-trained ControlNet DDPM
* **Model Comparison**: Tools to compare quality and speed across different approaches

## Model Types

### DDPM ControlNet (Baseline)
The original ControlNet implementation that uses a multi-step denoising process. Provides high-quality results but requires many inference steps (~1000 steps).

### Consistency ControlNet (Research Focus)
A **distilled version** of DDPM ControlNet that learns to predict clean images directly from noisy images in a **single step**. Uses consistency loss to ensure the model's predictions are consistent across different noise levels.

**Key Features:**
- Single-step sampling (1000x faster inference)
- Consistency loss for stable training
- Maintains control fidelity

### Distribution Matching ControlNet (Research Focus)  
A **distilled version** of DDPM ControlNet that uses distribution matching loss (KL divergence + MSE) to ensure the generated samples match the target distribution. Also provides **single-step sampling** for fast inference.

**Key Features:**
- Single-step sampling (1000x faster inference)
- Distribution matching loss for better sample quality
- Enhanced noise handling for pure-noise generation

## Setup
* Create a new conda environment with python 3.10 then run below commands
*  `conda activate <environment_name>`
* ```git clone https://github.com/explainingai-code/ControlNet-PyTorch.git```
* ```cd ControlNet-PyTorch```
* ```pip install -r requirements.txt```
* Download lpips weights by opening this link in browser(dont use cURL or wget) https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and downloading the raw file. Place the downloaded weights file in ```models/weights/v0.1/vgg.pth```
___  

## Data Preparation
### Mnist

For setting up the mnist dataset follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Ensure directory structure is following
```
ControlNet-PyTorch
    -> data
        -> mnist
            -> train
                -> images
                    -> *.png
            -> test
                -> images
                    -> *.png
```

### CelebHQ 
For setting up on CelebHQ, simply download the images from the official repo of CelebMASK HQ [here](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).

Ensure directory structure is the following
```
ControlNet-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg

```
---
## Configuration
 Allows you to play with different components of ddpm and autoencoder training
* ```config/mnist.yaml``` - Config for MNIST dataset
* ```config/celebhq.yaml``` - Configuration used for celebhq dataset

<ins>Relevant configuration parameters</ins>

Most parameters are self-explanatory but below I mention couple which are specific to this repo.
* ```autoencoder_acc_steps``` : For accumulating gradients if image size is too large for larger batch sizes
* ```save_latents``` : Enable this to save the latents , during inference of autoencoder. That way ddpm training will be faster

___  
## Training
The repo provides training and inference for Mnist(Unconditional DDPM) and CelebHQ (Unconditional LDM) and ControlNet with both these variations using canny edges.

For working on your own dataset:
* Create your own config and have the path in config point to images (look at `celebhq.yaml` for guidance)
* Create your own dataset class which will just collect all the filenames and return the image and its hint in its getitem method. Look at `mnist_dataset.py` or `celeb_dataset.py` for guidance 

Once the config and dataset is setup:
* For training and inference of Unconditional DDPM follow [this section](#training-unconditional-ddpm)
* For training and inference of ControlNet with Unconditional DDPM follow [this section](#training-controlnet-for-unconditional-ddpm)
* Train the auto encoder on your dataset using [this section](#training-autoencoder-for-ldm)
* For training and inference of Unconditional LDM follow [this section](#training-unconditional-ldm)
* For training and inference of ControlNet with Unconditional LDM follow [this section](#training-controlnet-for-unconditional-ldm)



## Training Unconditional DDPM
* For training ddpm on mnist,ensure the right path is mentioned in `mnist.yaml`
* For training ddpm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_ddpm.py#L40)
* For training DDPM run ```python -m tools.train_ddpm --config config/mnist.yaml``` for training ddpm with the desire config file
* For inference run ```python -m tools.sample_ddpm --config config/mnist.yaml``` for generating samples with right config file.

## Training ControlNet for Unconditional DDPM
* For training controlnet, ensure the right path is mentioned in `mnist.yaml`
* For training controlnet with ddpm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_ddpm_controlnet.py#L40)
* Ensure ```return_hints``` is passed as True in the dataset class initialization
* For training controlnet run ```python -m tools.train_ddpm_controlnet --config config/mnist.yaml``` for training controlnet ddpm with the desire config file
* For inference run ```python -m tools.sample_ddpm_controlnet --config config/mnist.yaml``` for generating ddpm samples using canny hints with right config file.


## Training AutoEncoder for LDM
* For training autoencoder on celebhq,ensure the right path is mentioned in `celebhq.yaml`
* For training autoencoder on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_vae.py#L49)
* For training autoencoder run ```python -m tools.train_vae --config config/celebhq.yaml``` for training autoencoder with the desire config file
* For inference make sure `save_latent` is `True` in the config
* For inference run ```python -m tools.infer_vae --config config/celebhq.yaml``` for generating reconstructions and saving latents with right config file.


## Training Unconditional LDM
Train the autoencoder first and setup dataset accordingly.

For training unconditional LDM ensure the right dataset is used in `train_ldm_vae.py` [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_ldm_vae.py#L43)
* ```python -m tools.train_ldm_vae --config config/celebhq.yaml``` for training unconditional ldm using right config
* ```python -m tools.sample_ldm_vae --config config/celebhq.yaml``` for generating images using trained ldm


## Training ControlNet for Unconditional LDM
* For training controlnet with celebhq, ensure the right path is mentioned in `celebhq.yaml`
* For training controlnet with ldm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Ensure Autoencoder and LDM have already been trained
* Call the desired dataset class in training file [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_ldm_controlnet.py#L43)
* Ensure ```return_hints``` is passed as True in the dataset class initialization
* Ensure ```down_sample_factor``` is correctly computed in the model initialization [here](https://github.com/explainingai-code/ControlNet-PyTorch/blob/main/tools/train_ldm_controlnet.py#L60)
* For training controlnet run ```python -m tools.train_ldm_controlnet --config config/celebhq.yaml``` for training controlnet ldm with the desire config file
* For inference with controlnet run ```python -m tools.sample_ldm_controlnet --config config/celebhq.yaml``` for generating ldm samples using canny hints with right config file.

## Training Controlnet Consistency Model from pre-trained Controlnet DDPM
* Train: ```python -m tools.train_consistency_controlnet_distilled --config config/mnist.yaml```
* Sample: ```python -m tools.sample_consistency_controlnet_distilled --config config/mnist.yaml --mode test --num_samples 10```

## Training Controlnet Distribution Matching Model from pre-trained Controlnet DDPM
* Train: ```python -m tools.train_distribution_matching_controlnet_distilled --config config/mnist.yaml```
* Sample: ```python -m tools.sample_distribution_matching_controlnet_distilled --config config/mnist.yaml --mode test --num_samples 10```

## Compare Models
* ```python -m tools.compare_controlnet_models --config config/mnist.yaml --num_samples 5 --ddpm_steps 1000```
* ```python -m tools.compare_all_controlnet_models --config config/mnist.yaml --num_samples 10 --ddpm_steps 50```

### Inference with Consistency Model
```bash
# Generate samples from test data
python -m tools.sample_consistency_controlnet_distilled --config config/mnist.yaml --mode test --num_samples 20

# Generate samples from random noise
python -m tools.sample_consistency_controlnet_distilled --config config/mnist.yaml --mode random --num_samples 10

# Generate samples from custom hints
python -m tools.sample_consistency_controlnet_distilled --config config/mnist.yaml --mode custom --num_samples 15
```

### Inference with Distribution Matching Model
```bash
# Generate samples from test data
python -m tools.sample_distribution_matching_controlnet_distilled --config config/mnist.yaml --mode test --num_samples 20

# Generate samples from random noise
python -m tools.sample_distribution_matching_controlnet_distilled --config config/mnist.yaml --mode random --num_samples 10

# Generate samples from custom hints
python -m tools.sample_distribution_matching_controlnet_distilled --config config/mnist.yaml --mode custom --num_samples 15
```

**Inference Modes**:
* **`test`**: Uses real MNIST images and their Canny edge hints
* **`random`**: Generates from random noise with random hints
* **`custom`**: Uses predefined geometric hints


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vae_autoencoder_samples```

During inference of autoencoder the following output will be saved
* Reconstructions for random images in  ```task_name```
* Latents will be save in ```task_name/vae_latent_dir_name``` if mentioned in config

During training and inference of unconditional ddpm or ldm following output will be saved:
* During training we will save the latest checkpoint in ```task_name``` directory
* During sampling, unconditional sampled image grid for all timesteps in ```task_name/samples/*.png``` . The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0

During training and inference of controlnet with ddpm/ldm following output will be saved:
* During training we will save the latest checkpoint in ```task_name``` directory
* During sampling, randomly selected hints and generated samples will be saved in ```task_name/hint.png``` and  ```task_name/controlnet_samples/*.png```. The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0



## Citations
```
@misc{zhang2023addingconditionalcontroltexttoimage,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2302.05543}, 
}
```

