# Deep Learning-Based Weather Downscaling

### Comparative Study of Deep Learning Models and an Attention-Enhanced U-Net Architecture

## Overview

This project investigates deep learning approaches for meteorological downscaling using ERA5 reanalysis data. The objective is to transform low-resolution 2-meter temperature fields into higher-resolution representations through deep learning.

Five teacher-provided baseline architectures were implemented and evaluated, followed by the development of a proposed U-Net architecture designed to improve downscaling performance.

## Project Objectives

- Reproduce and evaluate existing deep learning architectures.
- Develop a proposed U-Net model for weather downscaling.
- Compare model performance using RMSE and PSNR.
- Investigate the effectiveness of spatial feature extraction and attention mechanisms.

## Dataset

**Data source:** ERA5 Reanalysis (ECMWF)

**Target variable:** 2-meter temperature (t2m)

| Parameter | Description |
|---|---|
| Input resolution | 1° × 1° |
| Output resolution | 0.25° × 0.25° |
| Input dimensions | 20 × 35 |
| Output dimensions | 80 × 140 |
| Upscaling factor | 4× per dimension |
| Training samples | 11,688 |
| Validation samples | 2,920 |

Data normalization was performed using training-set statistics to avoid data leakage.

## Models

### 1. Baseline Models

The following implementations were provided as part of the course experiments:

- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Transformer
- Generative Adversarial Network (GAN)
- Diffusion Model

### 2. Proposed Model: Attention-Enhanced U-Net

An attention-enhanced U-Net architecture was developed as the proposed model.

**Main architectural components:**

- Convolutional encoder-decoder architecture
- Multi-level feature extraction
- Attention gate at the bottleneck
- Skip connections
- Transposed convolution for decoder upsampling
- PixelShuffle-based ×4 upsampling
- Batch Normalization and ReLU activation

The model was trained using PyTorch with the Adam optimizer and Mean Squared Error (MSE) loss.

## Experimental Results

The models were evaluated using Root Mean Square Error (RMSE) and Peak Signal-to-Noise Ratio (PSNR).

### Model Comparison

| Model | RMSE | PSNR (dB) |
|---|---:|---:|
| **U-Net (Proposed)** | **0.0500** | **39.03** |
| CNN | 0.0553 | 38.02 |
| GAN | ~0.0800 | ~31–33 |
| Diffusion | 0.1400 | 29.08 |
| LSTM | 0.2054 | 26.24 |
| Transformer | 1.4115 | 9.00 |

### Key Finding

The proposed U-Net achieved the best reported performance, with an RMSE of 0.0500.

Compared with the CNN baseline (RMSE = 0.0553), the proposed model achieved an approximately 9.6% reduction in RMSE.

### Results and Visualizations

The experimental visualizations are available in the [`results/`](results/) directory.

- [Model Comparison](results/model_comparison.png)
- [CNN Results](results/cnn_results2.png)
- [U-Net Results](results/unet_results.png)
- [LSTM Results](results/lstm_results.png)
- [Transformer Results](results/transformer_results.png)
- [GAN Results](results/gan_results.png)
- [Diffusion Results](results/diffusion_results.png)

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- ERA5 Reanalysis Data
- CUDA / GPU Computing

## Repository Structure

```text
Deep-Learning-Weather-Downscaling/
│
├── baseline_models/
│   └── (planned organization)
│
├── proposed_model/
│   └── (planned organization)
│
├── evaluation/
│   └── (planned organization)
│
├── results/
│   ├── model_comparison.png
│   ├── cnn_results2.png
│   ├── unet_results.png
│   ├── lstm_results.png
│   ├── transformer_results.png
│   ├── gan_results.png
│   └── diffusion_results.png
│
├── downscaling_cnn.py
├── downscaling_lstm.py
├── downscaling_transformer.py
├── downscaling_gan.py
├── downscaling_diffusion.py
├── unet_downscaling.py
├── DATA_GUIDE.md
└── README.md
```

## Limitations and Future Work

Potential future improvements include:

- Hyperparameter optimization.
- Ensemble U-Net architectures.
- Multi-variable meteorological downscaling.
- Physics-informed loss functions.
- Improved uncertainty quantification.
- Evaluation across additional geographical regions.

## Acknowledgements

The baseline model implementations were provided as part of the course experiments. The proposed U-Net architecture and comparative experimental study were developed as the project's main contribution.

**Project:** Comprehensive Practice in Smart Meteorology  
**Field:** Artificial Intelligence  
**Date:** June 2026
