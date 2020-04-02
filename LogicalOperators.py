# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:41:33 2019


@ Machine learning
"""
import numpy as np

def fun(h):
	
	if h >= 0:
		return 1
	else:
		return 0
	
def perceptron(x, w, b):
    
    h = np.dot(w, x) + b
    y = fun(h)
    return y


#Q2-Part a - implement the logical OR function
    
def OR_percep(x):
    w = np.array([1, 1])
    b = -1
    return perceptron(x, w, b)


    
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("OR({}, {}) = {}".format(1, 1, OR_percep(example1)))
print("OR({}, {}) = {}".format(1, 0, OR_percep(example2)))
print("OR({}, {}) = {}".format(0, 1, OR_percep(example3)))
print("OR({}, {}) = {}".format(0, 0, OR_percep(example4)))

##Q2-Part b - implement the logical AND function

def AND_percep(x):
    w = np.array([1, 1])
    b = -1.5
    
    return perceptron(x, w, b)


example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("AND({}, {}) = {}".format(1, 1, AND_percep(example1)))
print("AND({}, {}) = {}".format(1, 0, AND_percep(example2)))
print("AND({}, {}) = {}".format(0, 1, AND_percep(example3)))
print("AND({}, {}) = {}".format(0, 0, AND_percep(example4)))

##Q2-Part c - implement the logical XNOR function

def NOR_percep(x):
    w = np.array([-1, -1])
    b = 0.5

    return perceptron(x, w, b)

def XNOR_percep(x):
    gate_1 = AND_percep(x)
    gate_2 = NOR_percep(x)
    new_x = np.array([gate_1, gate_2])
    output = OR_percep(new_x)
    return output
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("XNOR({}, {}) = {}".format(1, 1, XNOR_percep(example1)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_percep(example2)))
print("XNOR({}, {}) = {}".format(0, 1, XNOR_percep(example3)))
print("XNOR({}, {}) = {}".format(0, 0, XNOR_percep(example4)))
