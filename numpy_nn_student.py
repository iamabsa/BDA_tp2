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

x_test_prepro = preprocess_input(x_test)
x_train_prepro = preprocess_input(x_train) 

# Task 4: output pre-processing
def preprocess_output(output):
    res = np.zeros((len(output), 10))
    res[np.arange(len(output)), output] = 1
    return res


y_test_prepro = preprocess_output(y_test)
y_train_prepro = preprocess_output(y_train)
        
# Task 5-6: creating and initializing matrices of weights

def layer_weights(m , n) :
    return np.random.standard_normal(size = (m , n)) / np.sqrt(n)

w1 = layer_weights(784 , 128)
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
    return [(input, a1, a2, softmax(z3)) , (z1, z2)]

#flattening the training data
testprep = preprocess_input(x_train)
print(testprep, testprep.shape, testprep.ndim)

input = testprep[0] #testing on the first image (first row)
#print(input, input.shape)
res = forward_pass_v2(input)
#print("showing res")
#print(res)

# Task 10: backpropagation

def back_propagation (forward_res, expected_output) :
    e3 = forward_res[0][-1] - expected_output

    deltaw3 = np.outer(np.transpose(forward_res[0][-2]) , e3)

    e2 = np.dot(e3 , np.transpose(w3)) * sigmoid_prime(forward_res[1][1])

    deltaw2 = np.outer(np.transpose(forward_res[0][-3]) , e2)

    e1 = np.dot(e2 , np.transpose(w2)) * sigmoid_prime(forward_res[1][0])

    deltaw1 = np.outer(np.transpose(forward_res[0][0]) , e1)

    return [deltaw1, deltaw2, deltaw3]

        
# Task 11: weight updates

def weight_updates(w1, w2, w3, back_res, learning_rate):

    w1 = w1 - learning_rate * back_res[0]
    w2 = w2 - learning_rate * back_res[1]
    w3 = w3 - learning_rate * back_res[2]
    return w1, w2, w3

# test11 = weight_updates(w1, w2, w3, back_propagation(forward_pass_v2(x_train_prepro[0]), y_train_prepro[0]) , 0.001) 
# print(w1.shape, test11[0].shape,w2.shape, test11[1].shape,w3.shape, test11[2].shape)       
# Task 12: computing error on test data

def error_rate(x,y):
     activation_arrays = np.array(forward_pass_v2(x)[0][3]).reshape(len(x),10)
     labels = np.argmax(activation_arrays, axis=1)
     labels = preprocess_output(labels)
     errors = np.any(labels != y, axis = 1)
     return np.sum(errors) / len(y)
        
# Task 13: error with initial weights

# test13 = error_rate(x_test_prepro, y_test_prepro)
# print("showing_test13")
# print(test13)
        
# Task 14-15: training

def train(number_epoch, learning_rate):
     w1 = layer_weights(784 , 128)
     w2 = layer_weights(128 , 64)
     w3 = layer_weights(64 , 10)
     for iter in range(number_epoch):
         start_time = time.time()
         index_output = 1
         for row in x_train_prepro[:10] :
             output = forward_pass_v2(row)
             delta_weights = back_propagation(output,y_train_prepro[index_output-1 :index_output])
             deltas = weight_updates(w1,w2,w3, delta_weights,learning_rate)
             w1 = deltas[0]
             w2 = deltas[1]
             w3 = deltas[2]
             index_output += 1
         print("epoch number %d : the error rate is %f , time : %f" %(iter +1 , error_rate(x_train_prepro, y_train_prepro), time.time() - start_time))        

train(10,0.001)          
# Task 16-18: batch training
