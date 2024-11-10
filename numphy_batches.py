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


def preprocess_input(input):
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


def layer_weights(m, n):
    return np.random.standard_normal(size=(m, n)) / np.sqrt(n)


w1 = layer_weights(784, 128)
w2 = layer_weights(128, 64)
w3 = layer_weights(64, 10)


def sigmoid(x):
    # Getting overflow error for large values of x
    # sigmoid(x) is 0 for x < -500 or 1 for x > 500
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # keepdims for broadcasting
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sigmoid_prime(x):
    x = np.clip(x, -500, 500)
    return np.exp(-x) / (np.exp(-x) + 1) ** 2


def forward_pass_v2(inputs):
    # Input is now a 2D array (batch of inputs)
    z1 = np.dot(inputs, w1)  # Shape: (batch_size, 128)
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2)  # Shape: (batch_size, 64)
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3)  # Shape: (batch_size, 10)
    output = softmax(z3)

    return [(inputs, a1, a2, output), (z1, z2)]


def back_propagation(forward_res, expected_output):
    # Expected_output and forward_res[0][-1] are both (batch_size, 10)
    e3 = forward_res[0][-1] - expected_output

    # forward_res[0][-2].T shape: (64, batch_size)
    # e3 shape: (batch_size, 10)
    deltaw3 = np.dot(forward_res[0][-2].T, e3)  # Shape: (64, 10)

    # Shape: (batch_size, 64)
    e2 = np.dot(e3, w3.T) * sigmoid_prime(forward_res[1][1])

    deltaw2 = np.dot(forward_res[0][-3].T, e2)  # Shape: (128, 64)

    # Shape: (batch_size, 128)
    e1 = np.dot(e2, w2.T) * sigmoid_prime(forward_res[1][0])

    deltaw1 = np.dot(forward_res[0][0].T, e1)  # Shape: (784, 128)

    return [deltaw1, deltaw2, deltaw3]

def weight_updates(back_res, learning_rate):
    global w1, w2, w3
    w1 = w1 - learning_rate * back_res[0]
    w2 = w2 - learning_rate * back_res[1]
    w3 = w3 - learning_rate * back_res[2]


def error_rate(x, y):
    activation_arrays = np.array(forward_pass_v2(x)[0][3])
    labels = np.argmax(activation_arrays, axis=1)
    labels = preprocess_output(labels)
    errors = np.any(labels != y, axis=1)
    return np.sum(errors) / len(y)


def train(number_epoch, learning_rate, batch_size=64):
    global w1, w2, w3
    n_samples = x_train_prepro.shape[0]
    for epoch in range(number_epoch):
        start_time = time.time()
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x_train_prepro[start_idx:end_idx]
            y_batch = y_train_prepro[start_idx:end_idx]
            output = forward_pass_v2(x_batch)
            delta_weights = back_propagation(output, y_batch)
            weight_updates(delta_weights, learning_rate)
        print("epoch number %d : the error rate is %f , time : %f" % (
            epoch + 1, error_rate(x_train_prepro, y_train_prepro), time.time() - start_time))


train(30, 0.001)
