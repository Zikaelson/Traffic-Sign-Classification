# Traffic Sign Classification Project

## Overview
This project involves building a Convolutional Neural Network (CNN) to classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model aims to help enhance the capabilities of autonomous vehicles by accurately recognizing and responding to various traffic signs.

## Dataset
The dataset used in this project is the German Traffic Sign Recognition Benchmark (GTSRB) dataset, available on Kaggle. It contains images of traffic signs with 43 different classes.

## Preprocessing

The dataset was downloaded from Kaggle and extracted.
## Image Resizing:

All images were resized to 50x50 pixels to ensure uniformity.
## Normalization:

The pixel values of the images were normalized to range between 0 and 1.
## Data Splitting:

The dataset was split into training and validation sets, with 80% of the data used for training and 20% for validation.
## One-Hot Encoding:

The class labels were converted to a one-hot encoded format for both the training and validation sets.
## Model Architecture
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture includes:

## Convolutional Layers:

Two convolutional layers with ReLU activation and max pooling.
## Dropout Layers:

Dropout layers to prevent overfitting.
## Flatten Layer:

A flatten layer to convert the 2D feature maps to a 1D vector.
## Dense Layers:

Fully connected dense layers with ReLU activation.
An output dense layer with softmax activation to produce probabilities for each class.
## Training
The model was compiled and trained using the following parameters:

Loss Function: Sparse categorical cross-entropy.
Optimizer: Adam.
Metrics: Accuracy.
Epochs: 50.
Batch Size: 128.
The training process included evaluating the model on the validation set after each epoch to monitor its performance.

## Evaluation
The trained model was evaluated using the test dataset. The test images were processed in the same way as the training images (resizing and normalization). The model's predictions were compared with the original labels to assess its accuracy.

## Results
The model achieved an accuracy of over 95% on the validation set within 50 epochs. The accuracy and loss over the epochs were visualized to understand the model's learning curve.

## Future Work
To further improve the model's performance, the following steps can be taken:
Hyperparameter Tuning: Adjusting parameters like learning rate, batch size, and network architecture.
Data Augmentation: Applying transformations to the images to increase the diversity of the training set.
Ensemble Methods: Combining multiple models to improve overall performance.
