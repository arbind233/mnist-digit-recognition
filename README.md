# Handwritten Digit Recognition (MNIST)

## Problem
Computers need to read handwritten digits (0-9) for tasks like cheque, form and postal code processing. Handwriting varies a lot between people, so fixed rules don't work well.

## Solution
A Convolutional Neural Network (CNN) built with TensorFlow/Keras and trained on MNIST (70,000 grayscale 28x28 images). The model uses data augmentation (random rotation and zoom), batch normalization and dropout.

**Test accuracy: 98.92%**

Model: Conv2D(32) → BatchNorm → MaxPool → Conv2D(64) → BatchNorm → MaxPool → Flatten → Dropout(0.4) → Dense(10, softmax)

## Setup steps
1. Open `MNIST_CNN.ipynb` in Google Colab
2. Run all cells in order (training takes about 3-5 minutes)
3. Run `app.py` code in a new cell to launch the Gradio app
4. Draw a digit in the box to see the prediction

Install locally: `pip install -r requirements.txt`
Run tests: `pytest`
