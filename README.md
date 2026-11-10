# Handwritten Digit Recognition (MNIST)

## Problem
Computers need to read handwritten digits (0-9) for tasks like cheque, form and postal code processing. Handwriting varies a lot between people, so fixed rules don't work well.

## Solution
A Convolutional Neural Network (CNN) built with TensorFlow/Keras and trained on MNIST (70,000 grayscale 28x28 images). The model uses data augmentation (random rotation and zoom), batch normalization and dropout.

**Test accuracy: 98.92%**

Model: Conv2D(32) → BatchNorm → MaxPool → Conv2D(64) → BatchNorm → MaxPool → Flatten → Dropout(0.4) → Dense(10, softmax)
