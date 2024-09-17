# MNIST Handwritten Digit Classifier

An implementation of a multilayer neural network for recognizing handwritten digits using Keras and TensorFlow. This project achieves an accuracy of 98.31% using Keras and over 99% using TensorFlow.

## About the MNIST Dataset
The MNIST (Modified National Institute of Standards and Technology) database consists of 70,000 examples of handwritten digits (0–9) in grayscale images, where 60,000 are used for training and 10,000 for testing. The images are normalized to fit within a 28x28 pixel box, with grayscale levels introduced via anti-aliasing.

## Structure of the Neural Network
The network is a feedforward architecture consisting of the following layers:
- **Input Layer:** Takes in the 28x28 pixel input images.
- **Hidden Layers:** These unobservable layers process the input through neuron activations. The number of hidden layers and their activation functions are configurable.
- **Output Layer:** Produces the final digit classification, with each output corresponding to one of the digits (0–9).

The neural network is trained using backpropagation, with the weights and biases of the neurons being adjusted to minimize the prediction error.

### Neural Network Diagram:
![Small Labelled Neural Network](http://i.imgur.com/HdfentB.png)

### Model Summary:
![Model Summary](https://github.com/aakashjhawar/handwritten-digit-recognition/blob/master/assets/model/model_summary.png)

## Getting Started

### Prerequisites
- **Python 3.5 or higher**
- **Keras** for building the model.
- **TensorFlow** as the backend for Keras.
- **OpenCV** for image handling.
  Install OpenCV:
  ```bash
  sudo apt-get install python-opencv

