# Multi-Scale Convolutional Neural Networks for Speech Emotion Recognition with Dynamic Temporal Feature Learning

## Overview

Welcome to the Speech Emotion Recognition (SER) project repository! This project aims to detect emotions from speech using deep neural networks and meanwhile explore the impact of gender and noise on the performance of SER.

## Introduction

Speech Emotion Recognition (SER) is the process of identifying emotions expressed in speech signals using machine learning algorithms. In this project, I proposed a novel approach leveraging multi-scale convolutional neural networks to dynamically learn temporal features from Mel-frequency cepstral coefficient (MFCC) spectrograms extracted from speech signals. The model integrates multi-scale time information to capture temporal variations within speech more effectively, resulting in impressive performance on the RAVDESS dataset. Additionally, the impact of noise and gender on speech emotion recognition was investigated, providing insights into enhancing model robustness and the potential of gender-specific classifiers.

## Network Architecture

Below is the overall architecture of the Multi-scale Convolutional Neural Networks:

<img width="922" alt="network-architecture" src="https://github.com/timchan9742/Multi-scale-Convolutional-Neural-Network-for-Speech-Emotion-Recognition/assets/167204379/366f89f1-ac9e-42c4-a328-20d8caa0ded4">
<img width="762" alt="cnn-block" src="https://github.com/timchan9742/Multi-scale-Convolutional-Neural-Network-for-Speech-Emotion-Recognition/assets/167204379/e035ea77-6950-4194-8eb9-2b6867dd0eb8">


## Project Structure

- **/data**: This directory contains the preprocessed data used in the project. 
- **/models**: This directory contains all the trained models generated during the training phase of the project. These models can be used for inference on new data.
- **/RAVDESS**: This directory contains the original audio files from the RAVDESS dataset, the dataset is also available on [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
- **data_preprocessing.ipynb**: This notebook provides detailed steps for preprocessing the raw audio data, including feature extraction and data augmentation techniques.
- **model_training_testing.ipynb**: This notebook contains the code for model training and evaluation. 

## Results

<img width="663" alt="cm-all" src="https://github.com/timchan9742/Multi-scale-Convolutional-Neural-Network-for-Speech-Emotion-Recognition/assets/167204379/cc652fb2-56ca-4200-a74b-1c586e916c29">
<br/><br/>

The multi-scale dilated convolutional neural networks demonstrate promising results in SER, the gender-independent classifier achieved high evaluation scores of 90% on Weighted Accuracy Rate (WAR) and 89.8% on Unweighted Accuracy Rate (UAR). Besides, incorporating noise into the training data as a form of data augmentation mitigated imbalanced data issues and notably enhanced recognition performance, leading to more robust results. Additionally, the study highlighted the significant influence of gender in SER, with emotions being more discernible in female speech compared to male speech.

## Requirements

To run the code in this repository, you'll need the following Python libraries:

- numpy: For numerical computing and array manipulation.
- matplotlib: For data visualization and plotting.
- librosa: For audio feature extraction.
- tensorflow: For building and training neural networks.
- keras: A high-level neural networks API (included with TensorFlow).
- pickle: For serializing and deserializing Python objects.
