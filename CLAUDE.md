# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a TensorFlow.js image classification model for waste sorting, built with Google Teachable Machine. The model classifies images into three categories: Paper, Plastic, and Biodegradable materials.

## Model Architecture

- **Framework**: TensorFlow.js 1.7.4 with Teachable Machine 2.4.12
- **Model Type**: Image classification using MobileNet-like architecture
- **Input**: 224x224 RGB images
- **Classes**: 3 categories (Paper, Plastic, Biodegradable)
- **Architecture**: Sequential model with depthwise separable convolutions, batch normalization, and residual connections

## File Structure

- `model/model.json`: Model topology and architecture (large JSON file)
- `model/weights.bin`: Pre-trained model weights
- `model/metadata.json`: Model metadata including labels, version info, and configuration

## Key Model Components

The model uses a MobileNetV2-style architecture with:
- Zero-padding layers for input processing
- Depthwise separable convolutions (Conv2D + DepthwiseConv2D)
- Batch normalization layers (BN) with momentum=0.999, epsilon=0.001
- ReLU activations with max_value=6
- Residual connections (Add layers) for gradient flow
- Progressive downsampling through strided convolutions

## Working with This Model

When integrating or modifying this model:
1. The model expects 224x224x3 input tensors
2. Use TensorFlow.js compatible preprocessing for input images
3. Model outputs softmax probabilities for 3 classes
4. Weights file must be loaded alongside model.json for inference
5. Compatible with @teachablemachine/image package version 0.8.4-alpha2