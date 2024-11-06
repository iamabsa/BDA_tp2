#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: DIAGNE Mame Absa and Giacometti Luca
"""

import struct
import numpy as np
import math as mt
import time
import matplotlib.pyplot as plt


# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    


# Task 1: reading the MNIST files into Python ndarrays

x_test = read_idx("t10k-images.idx3-ubyte")

y_test = read_idx("t10k-labels.idx1-ubyte")

x_train = read_idx("train-images.idx3-ubyte")

y_train = read_idx("train-labels.idx1-ubyte")

print(x_train)
print(y_train)
print(x_test)
print(y_test)

print(x_test.shape, x_test.size, x_test.ndim, y_test.shape)
print()

#Question 1 : the shape of the x_test array is (10000, 28, 28). The first represents the number of digits or instance
#and the last one represent the bitmap of one digit. Each row represents one digit
        
# Task 2: visualize a few bitmap images

firstattempt = x_test[0][0]
print(firstattempt, firstattempt.shape)
# plt.imshow(firstattempt)
# plt.show()
        
# Task 3: input pre-preprocessing  

def preprocess_input(input) :
    input = input / 256
    min_val = np.min(input)
    max_val = np.max(input)
    scaled_input = (input - min_val) / (max_val - min_val)
    shape = input.shape
    firstdim = shape[0]
    secondim = shape[1] * shape[2]
    return scaled_input.reshape(firstdim, secondim)
  

# Task 4: output pre-processing
def preprocess_output(output):
    res = np.zeros((len(output), 10))
    res[np.arange(len(output)), output] = 1
    return res
        
# Task 5-6: creating and initializing matrices of weights

def layer_weights(m , n) :
    return np.random.standard_normal(size = (m , n)) / np.sqrt(n)

w1 = layer_weights(784 , 128)
#print(w1.shape)
w2 = layer_weights(128 , 64)
w3 = layer_weights(64 , 10)
        
# Task 7: defining functions sigmoid, softmax, and sigmoid'

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    total_sum = np.sum(exp_x)
    return exp_x / total_sum

def sigmoid_prime(x) :
    return np.exp(-x) / (np.exp(-x) + 1 ) ** 2
        
# Task 8-9: forward pass

def forward_pass(input) :
    z1 = np.dot(input , w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1 , w2)
    a2 = sigmoid(z2)
    z3 = np.dot(a2 , w3)
    return [sigmoid(z3)]

def forward_pass_v2(input) :
    z1 = np.dot(input , w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1 , w2)
    a2 = sigmoid(z2)
    z3 = np.dot(a2 , w3)
    return [(input, a1, a2, sigmoid(z3)) , (z1, z2)]

#flattening the training data
testprep = preprocess_input(x_train)
print(testprep, testprep.shape, testprep.ndim)

input = testprep[0] #testing on the first image (first row)
print(input, input.shape)
res = forward_pass_v2(input)
print("showing res")
#print(res)

# Task 10: backpropagation

def backpropagation (forward_res, expected_output) :
    e3 = forward_res[0][-1] - expected_output

    deltaw3 = np.outer(np.transpose(forward_res[0][-2]) , e3)

    e2 = np.dot(e3 , np.transpose(w3)) * sigmoid_prime(forward_res[1][1])

    deltaw2 = np.outer(np.transpose(forward_res[0][-3]) , e2)

    e1 = np.dot(e2 , np.transpose(w2)) * sigmoid_prime(forward_res[1][0])

    deltaw1 = np.outer(np.transpose(forward_res[0][0]) , e1)

    return [deltaw1, deltaw2, deltaw3]

test10  = backpropagation(forward_pass_v2(input), ([0,0,0,0,0,0,0,1,0,0]))
print("showing what is inside test 10")
print(test10[0].shape, test10[1].shape, test10[2].shape, len(test10))
print(test10)
        
# Task 11: weight updates

def weight_updates(w1, w2, w3,back_res):
    w1 = w1 - 0.001 * back_res[0]
    w2 = w2 - 0.001 * back_res[1]
    w3 = w3 - 0.001 * back_res[2]
    return w1, w2, w3
        
# Task 12: computing error on test data
        
# Task 13: error with initial weights
        
# Task 14-15: training

# Task 16-18: batch training
