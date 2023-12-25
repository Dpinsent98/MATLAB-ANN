Artificial Neural Network (ANN) for MNIST Classification

Overview
This repository contains MATLAB code implementing an Artificial Neural Network (ANN) for classifying handwritten digits from the MNIST dataset. 
The code is designed for educational purposes and follows principles outlined in the book "Make Your Own Neural Network" by Tariq Rashid. 
The rev file implements a reverse query in order to see inside the mind of a neural network.

Features
Configurable neural network parameters, including the number of hidden layers, nodes in each layer, learning rate, and more.
Supports heuristic unscaled, heuristic scaled, and calculus-based backpropagation methods.
Choice of performance index: total squared error or cross-entropy.
Visualization of intermediate graphics during training for better understanding.

Usage
Requirements
MATLAB environment
MNIST dataset files (mnist_train_1000.csv and mnist_test_100.csv) in the same folder as the MATLAB code.
Running the Code
matlab
Copy code
[Yn On Yt Ot] = ann2050236(2050236, 50, 0.025, 1, 10, 1, 1, 1, 1, 1);
Adjust the input parameters as needed. The function returns exact and predicted labels for both training and testing sets.

Configuration Options
Nh: Number of hidden layers (1 or 2)
bp: Backpropagation method (1, 2, or 3)
cf: Performance index choice (1 or 2)
gfx: Display intermediate graphics (1 or 2)

Acknowledgments
Thanks to Tariq Rashid for the educational content provided in the book.

