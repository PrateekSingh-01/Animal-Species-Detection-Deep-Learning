# 🐾 Animal Species Detection

A deep learning application for classifying animal images
into 90 different species using transfer learning with ResNet18.

## Features

- 90 animal species
- PyTorch
- ResNet18 transfer learning
- Fine-tuning
- GPU training
- Streamlit interface
- Single-image prediction
- Confidence score

## Dataset

5,400 images
90 classes
60 images per class

Train: 4,320
Validation: 540
Test: 540

## Models

Custom CNN
Test Accuracy: 20.93%

ResNet18 Feature Extraction
Test Accuracy: 85.93%

ResNet18 Fine-Tuning
Test Accuracy: 92.78%

## Tech Stack

Python
PyTorch
Torchvision
Scikit-learn
Pillow
Streamlit

## Architecture

Image
 ↓
Resize 224×224
 ↓
Normalization
 ↓
ResNet18
 ↓
Fine-tuned feature extractor
 ↓
90-class classifier
 ↓
Animal prediction