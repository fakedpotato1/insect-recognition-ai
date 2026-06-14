import numpy as np

def matmul(A, B):
    return np.dot(np.array(A), np.array(B))

def element_wise_op(A, B, op):
    arr_A, arr_B = np.array(A), np.array(B)
    if op == '+': return arr_A + arr_B
    elif op == '-': return arr_A - arr_B
    elif op == '*': return arr_A * arr_B
    else: raise ValueError("Unsupported operation")

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))

def relu(x):
    return np.maximum(0, np.array(x))