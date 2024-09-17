# Handwritten Digit Classifier

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

### Model Summary:

## Getting Started

### Prerequisites
- **Python 3.5 or higher**
- **Keras** for building the model.
- **TensorFlow** as the backend for Keras.
- **OpenCV** for image handling.
  Install OpenCV:
  ```bash
  sudo apt-get install python-opencv
Installation
Clone this repository:

git clone https://github.com/Yarra-Hemanth/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition

Install the required dependencies:
pip3 install -r requirements.txt

Running the Model
To train the model using TensorFlow:
python3 tf_cnn.py

To skip training and load a pre-trained model:
python3 load_model.py <path/to/image_file>

For example:
python3 load_model.py assets/images/1a.jpg
Saving and Loading Model Weights
Save Model Weights After Training

You can save the trained model weights by running the following:
python CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5

or

python3 CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5

Load Saved Model Weights
To load the saved model weights and avoid retraining:
python CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5

or

python3 CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5

Accuracy Results
Machine Learning Algorithms:
K Nearest Neighbors: 96.67%
SVM: 97.91%
Random Forest Classifier: 96.82%
Deep Neural Networks:
Three Layer CNN (TensorFlow): 99.70%
Three Layer CNN (Keras & Theano): 98.75%
All code is written in Python 3.5, and executed on an Intel Xeon Processor / AWS EC2 Server.

Test Images Classification Output:
  
Result:

The following image shows the prediction from the CNN model:


This README file includes all the instructions and details needed for anyone looking to clone and use the project. It covers installation, running the model, saving/loading weights, and viewing results.
