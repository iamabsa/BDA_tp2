#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: XXX and YYY
"""

import struct
import numpy as np
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

firstattempt = x_test[0]
#print(firstattempt, firstattempt.shape)
plt.imshow(firstattempt)
plt.show()
        
# Task 3: input pre-preprocessing    

# Task 4: output pre-processing
        
# Task 5-6: creating and initializing matrices of weights
        
# Task 7: defining functions sigmoid, softmax, and sigmoid'
        
# Task 8-9: forward pass
        
# Task 10: backpropagation
        
# Task 11: weight updates
        
# Task 12: computing error on test data
        
# Task 13: error with initial weights
        
# Task 14-15: training

# Task 16-18: batch training
