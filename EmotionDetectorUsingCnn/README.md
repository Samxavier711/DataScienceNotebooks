# Face Expression Recognition with Keras and OpenCV

This project demonstrates the creation of a Sequential Convolutional Neural Network (CNN) model using Keras to detect and classify facial expressions. The model is trained on the Kaggle Facial Expression Recognition Challenge dataset, and it uses techniques like early stopping and a learning rate scheduler for efficient model training.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Real-time Emotion Detection](#real-time-emotion-detection)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Project Overview

The goal of this project is to recognize facial expressions in real-time using a trained CNN model. The model is created with Keras and trained on a Kaggle dataset containing labeled facial expressions.

## Dataset

We used the dataset - (link) (https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset)). This dataset contains seven classes of emotions, including anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Model Architecture

Our CNN model architecture consists of several convolutional layers, max-pooling layers, and fully connected layers. The model is designed to effectively capture and classify facial expressions.

## Training

We trained the model using techniques such as early stopping to prevent overfitting and a learning rate scheduler to fine-tune the learning process. The training parameters and details can be found in the code.

## Real-time Emotion Detection

To use the trained model for real-time emotion detection, we have included a `recorder.py` script. This script accesses your computer's camera using OpenCV and utilizes a Haar classifier to detect faces in real-time. Once a face is detected, it feeds the face image as input to the trained model, and the predicted emotion is displayed in real-time.

## Dependencies

- Python 3.x
- Keras
- OpenCV


## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Samxavier711/DataScienceNotebooks.git
2. Change the path for the model.h5 and haarClassifier.xml file accordingly.
3. Run the Recorder.py file.
