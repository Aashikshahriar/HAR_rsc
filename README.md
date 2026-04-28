# HAR_rsc
Lightweight Privacy-Preserving Human Activity Recognition from CSI Data using CNN–Temporal Attention Network
Overview

This repository contains the implementation and experimental pipeline supporting the research work:

“Lightweight Privacy-Preserving Human Activity Recognition from CSI Data using CNN-Temporal Attention Network.”

The work focuses on WiFi Channel State Information (CSI)-based Human Activity Recognition (HAR) and introduces a privacy-preserving deep learning framework that integrates Convolutional Neural Networks (CNN) with a Temporal Attention mechanism, while incorporating Differential Privacy (DP) to protect sensitive behavioral patterns.

Research Objective

The primary objectives of this research are:

Develop a lightweight deep learning architecture capable of accurately recognizing human activities using WiFi CSI signals.

Integrate temporal attention mechanisms to capture temporal dependencies in CSI signals and improve recognition performance.

Incorporate Differential Privacy (DP) into the training pipeline to provide formal privacy guarantees for CSI-based HAR systems.

Evaluate privacy–utility trade-offs by analyzing recognition performance across different privacy budgets (ε).

Validate the framework across multiple CSI datasets under varying environmental conditions such as distance, height, and sensing scenarios.

The repository includes training pipelines, preprocessing modules, differential privacy experiments, and evaluation scripts used in the study.

This repository contains the codes and data supporting the research "Lightweight Privacy-Preserving Human Activity Recognition from CSI Data using CNN-Temporal Attention Network". This project contains work on three publicly available dataset CSI-HAR, CSLOS and Wi-AR. Differential Privacy is incorporated for making it privacy aware detection.
The dataset can be found in the following links: 
1. CSI-HAR Dataset: https://github.com/parisafm/CSI-HAR-Dataset
2. CSLOS Dataset: https://data.mendeley.com/datasets/v38wjmz6f6/1
3. WiAR Dataset: https://github.com/linteresa/WiAR

The repository contains several modules that collectively implement the proposed CNN–Temporal Attention framework for CSI-based Human Activity Recognition (HAR). The model.py file defines the core architecture consisting of 1D convolutional layers for spatial feature extraction, a temporal attention module for capturing sequential dependencies, and fully connected layers for final activity classification. The train.py script manages the training pipeline, including loading the preprocessed datasets, optimizing the model parameters, and saving the best-performing model. The evaluate.py script is used to assess model performance on unseen test data and reports key evaluation metrics such as accuracy and classification statistics. To incorporate privacy guarantees, dp_experiment.py implements Differential Privacy training using the Opacus framework, enabling ε-based privacy budget experiments and analysis of the privacy–utility trade-off. Additionally, accvssample.py evaluates sample efficiency by analyzing recognition accuracy with varying training sample sizes. The repository also includes dataset-specific preprocessing scripts such as Preprocessing_CSI_HAR, preprocessing_code_CSLOS.py, distance_preprocessing.py, and height_preprocessing.py, which prepare CSI signals through normalization, segmentation, and window slicing for different experimental conditions. Finally, model_cnn_attention.pdf presents the architectural diagram of the proposed CNN–Temporal Attention model used in this research.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt


In addition to the provided preprocessing scripts for CSI-HAR, CSLOS, and WiAR datasets, this repository supports the use of custom or externally preprocessed datasets. This allows researchers to adapt the framework to new CSI datasets or other time-series sensing modalities.

To use a custom dataset, the data must be converted into a standardized format compatible with the training pipeline. Specifically, the input features should be stored as a NumPy array X.npy with shape (N, T, C), where N represents the number of samples, T denotes the number of time steps, and C corresponds to the feature dimension (e.g., subcarriers or antenna channels). The labels should be stored as y.npy with shape (N,) and must be integer-encoded.

Before training, users should ensure that the data has undergone appropriate preprocessing steps. These typically include normalization (such as per-sample z-score normalization), segmentation of continuous CSI streams into fixed-length windows, and label encoding. Optional augmentation techniques such as Gaussian noise injection, temporal jittering, or amplitude scaling may also be applied to improve model robustness.

Once the dataset is prepared, users can integrate it into the pipeline by modifying the data loading step in the preprocessing module or directly within the training script. The model expects consistent input dimensions across all samples, and the same preprocessing procedure should be applied during both training and evaluation to ensure reliable results.

This flexible design allows the framework to be extended beyond the included datasets, making it suitable for a wide range of applications such as indoor sensing, wearable signal analysis, or other privacy-sensitive time-series recognition tasks.
