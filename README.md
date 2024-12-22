 Hand Gesture Recognition System

This repository contains the code and resources to develop a deep learning-based model capable of identifying and classifying various hand gestures from images or video data. The aim is to enable intuitive human-computer interactions and gesture-based control systems.

## Table of Contents

- [Overview]
- [Features]
- [Tech Stack]
- [Data Requirements]
- [Model Architecture]
- [Setup Instructions]
- [Usage]
- [Contributing]

## Overview

### Problem Statement

Develop a model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems.

### Context

This project uses a **hand gesture recognition database** consisting of near-infrared images captured by the **Leap Motion sensor**. The database includes:

- 10 distinct hand gestures.
- Performed by 10 different subjects (5 men and 5 women).

## Features

- **Gesture Recognition**: Accurately identifies hand gestures from input images.
- **Multi-Subject Training**: Incorporates diverse gesture styles for robustness.
- **Deep Learning Techniques**: Leverages CNNs for feature extraction and classification.
- **Extensible Framework**: Easily adaptable for additional gestures or sensors.

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: Keras with TensorFlow backend
- **Image Processing**: OpenCV
- **Other Tools**: NumPy, Matplotlib

## Data Requirements

- **Dataset**: Near-infrared hand gesture images captured by the Leap Motion sensor.
- **Categories**: 10 distinct hand gestures.
- **Subjects**: Performed by 10 individuals (5 male, 5 female).
- **Data Augmentation**: Techniques applied to enhance dataset diversity.

### Dataset Setup

Ensure the dataset is structured into training and testing directories, with subfolders for each gesture category.

## Model Architecture

The model employs Convolutional Neural Networks (CNNs) for effective feature extraction and classification. Key components include:

1. **Convolutional Layers**: For spatial feature extraction.
2. **Max-Pooling Layers**: For dimensionality reduction.
3. **Fully Connected Layers**: For classification.
4. **Dropout Layers**: To prevent overfitting.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd hand-gesture-recognition
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare the dataset:
   - Place the dataset in the `data/` directory.
   - Ensure training and testing splits are correctly organized.

5. Verify environment setup:
   ```bash
   python -c "import keras; print(keras.__version__)"
   ```

## Usage

1. Train the model:
   ```bash
   python train.py
   ```
2. Test the model with sample images:
   ```bash
   python test.py --image <path_to_image>
   ```
3. Evaluate performance:
   ```bash
   python evaluate.py
   ```

## Contributing

We welcome contributions from the community! If you would like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations.
