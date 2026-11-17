# Deep Learning-Based Weather Downscaling

### Comparative Study of Deep Learning Models with a Proposed Attention-Enhanced U-Net

## Overview

This project explores deep learning approaches for weather data downscaling, focusing on 2-meter temperature (T2M).

The work was conducted in two stages:

1. **Baseline experimentation:** Running and evaluating teacher-provided deep learning implementations, including CNN, LSTM, Transformer, GAN, and Diffusion models.
2. **Proposed model development:** Designing and implementing an attention-enhanced U-Net architecture to improve the downscaling performance of the baseline approaches.

The main objective was to investigate different neural network architectures and develop a model capable of achieving better reported performance than the baseline models.

## Project Objectives

- Reproduce and experiment with existing deep learning architectures.
- Develop a U-Net-based weather downscaling model.
- Compare model performance using Root Mean Square Error (RMSE).
- Explore attention mechanisms and encoder-decoder architectures.
- Analyze the strengths and limitations of different deep learning approaches.

## Dataset and Preprocessing

The project is designed around ERA5 reanalysis temperature data from the European Centre for Medium-Range Weather Forecasts (ECMWF).

**Target variable:** 2-meter temperature (T2M).

The accompanying data guide describes the following intended configuration:

| Parameter | Description |
|---|---|
| Low-resolution input | 1.0° × 1.0° |
| High-resolution target | 0.25° × 0.25° |
| Temporal resolution | 6 hours |
| Geographic region | China |
| Intended period | 2010–2020 |
| Upscaling factor | 4× per spatial dimension |

The U-Net implementation loads preprocessed low-resolution and high-resolution arrays from the `processed_era5_t2m_downscaling` directory.

**Note:** The dataset files are not included in this repository. Refer to `DATA_GUIDE.md` for the data preparation instructions. The actual data provenance and preprocessing configuration used for each reported experiment should be verified before treating the results as a fully reproducible ERA5 benchmark.

## Models

### 1. Baseline Models

The following implementations were provided as part of the course assignment and used for comparative experiments:

| Model | Architecture |
|---|---|
| CNN | Convolutional Neural Network |
| LSTM | Long Short-Term Memory |
| Transformer | Attention-based neural network |
| GAN | Generative Adversarial Network |
| Diffusion | Diffusion-based generative model |

### 2. Proposed Model: Attention-Enhanced U-Net

The proposed model is a U-Net-based architecture designed for spatial temperature downscaling.

Its main components include:

- **Encoder:** Three convolutional feature extraction stages with 32, 64, and 128 channels.
- **Bottleneck:** A convolutional block with 256 channels.
- **Attention mechanism:** A lightweight attention gate applied to bottleneck features.
- **Skip connections:** Combine encoder and decoder features to preserve spatial information.
- **Decoder:** Transposed convolution and convolutional blocks.
- **Upsampling:** Two PixelShuffle operations, each providing 2× upsampling.
- **Output layer:** A final convolution producing a single-channel temperature prediction.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Learning rate | 0.0001 |
| Batch size | 8 |
| Training epochs | 50 |
| Hardware | CUDA-enabled GPU |

## Experimental Results

The following RMSE values were reported in the experiment logs and model comparison output.

### Baseline and Proposed Model Comparison

| Model | Reported RMSE |
|---|---:|
| **U-Net (Proposed)** | **0.0500** |
| CNN | 0.0553 |
| Diffusion | 0.1400 |
| LSTM | 0.2054 |
| Transformer | 1.4115 |
| GAN | Not reported |

**Important:** These are preliminary reported results, not yet a fully reproducible benchmark. The current evaluation script contains hardcoded values for several models and needs further work before independently reproducing the comparison.

### Key Finding

The proposed U-Net achieved the lowest reported RMSE of **0.0500**.

Compared with the teacher-provided CNN baseline (0.0553), this represents an approximately 9.6% reduction in reported RMSE.

Further validation under a consistent evaluation procedure is needed to confirm the comparison.

## Results and Experimental Logs

The `results/` directory contains the experimental screenshots and comparison output.

| Experiment | Results |
|---|---|
| Model comparison | [View results](results/model_comparison.png) |
| CNN | [View results](results/cnn_results2.png) |
| U-Net | [View results](results/unet_results.png) |
| LSTM | [View results](results/lstm_results.png) |
| Transformer | [View results](results/transformer_results.png) |
| GAN | [View results](results/gan_results.png) |
| Diffusion | [View results](results/diffusion_results.png) |

These images document the training and evaluation logs rather than serving as spatial prediction visualizations.

## Repository Structure

```text
Deep-Learning-Weather-Downscaling/
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
├── download_data.py
├── DATA_GUIDE.md
├── requirements.txt
├── .gitignore
└── README.md


## Technologies

- Python
- PyTorch
- NumPy
- CUDA
- Matplotlib
- ERA5 Reanalysis Data

## Limitations and Future Work

Potential improvements include:

- Developing a reproducible evaluation pipeline for all architectures.
- Ensuring identical validation data and evaluation procedures across models.
- Investigating different loss functions and optimization strategies.
- Improving attention mechanisms.
- Generating and comparing spatial temperature prediction maps.
- Evaluating model generalization across different geographical regions.
- Exploring uncertainty estimation and physics-informed approaches.

## Acknowledgements

The baseline model implementations were provided by the course instructor as part of the practical assignment.

The proposed U-Net implementation, training experiments, and comparative analysis were conducted as part of the project work.

**Project:** Comprehensive Practice in Smart Meteorology  
**Field:** Artificial Intelligence  
**Focus:** Deep Learning and Weather Downscaling
