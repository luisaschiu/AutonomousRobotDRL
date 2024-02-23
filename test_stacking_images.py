import numpy as np
from collections import deque
import tensorflow as tf
import cv2 as cv

arr1 = np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]])
arr2 = np.array(
        [[2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2]])
arr3 = np.array(
        [[3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3]])
arr4 = np.array(
        [[4, 4, 4, 4],
         [4, 4, 4, 4],
         [4, 4, 4, 4],
         [4, 4, 4, 4]])
arr5 = np.array(
        [[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]])
# Testing expected output using np.stack method:
# NOTE: I am not going with this approach, since using deque would be easier to handle keeping a maximum of 4 sequential time frames.
# This is to show the expected result for the methods using deque.
print("arr1 shape:", str(arr1.shape))
stacked_arr = np.stack((arr1, arr2, arr3), axis=-1, dtype=np.float32)
print("Stacked array/Expected Result: ")
print( stacked_arr)
print("Stacked array/Expected Result shape: ")
print(stacked_arr.shape)

# Using deque: stack images by appending to deque object type, transposes to expected result shape
# NOTE: For now, I am going with the tensorflow object type since it makes more sense to me, inputting a tensor into the neural network

# Method 1: Convert to tensorflow object type
replay_memory = deque(maxlen=3)
cur_stacked_images = deque(maxlen = 4)
cur_stacked_images.append(arr1)
cur_stacked_images.append(arr2)
cur_stacked_images.append(arr3)
cur_stacked_images.append(arr4)
# Assign to state variable, similar to what we will do in class_DQN
state = cur_stacked_images
print("Before converting to tensor: ")
print(state)
tensor = tf.constant(cur_stacked_images, tf.float32)
print("After converting to tensor: ")
print(tensor)
tensor_transposed = tf.transpose(tensor, [1, 2, 0])
print("tensor_ transposed: ")
print(tensor_transposed)
print("Add batchsize dimension: ")
batchsize = tf.expand_dims(tensor_transposed, axis=0)
replay_memory.append(batchsize)
print(batchsize)

# # Test new timestep:
# print("Adding timestep 5: ")
# cur_stacked_images.append(arr5)
# tensor = tf.constant(cur_stacked_images, tf.float32)
# print("After converting to tensor: ")
# print(tensor)
# tensor_transposed = tf.transpose(tensor, [1, 2, 0])
# print("tensor_ transposed: ")
# print(tensor_transposed)
# print("Add batchsize dimension: ")
# batchsize = tf.expand_dims(tensor_transposed, axis=0)
# print(batchsize)

# # Method 2: Convert to numpy array object type
# cur_stacked_images = deque(maxlen = 4)
# cur_stacked_images.append(arr1)
# cur_stacked_images.append(arr2)
# cur_stacked_images.append(arr3)
# # Assign to state variable, similar to what we will do in class_DQN
# state = np.array(cur_stacked_images)
# print("Before converting to numpy array: ")
# print(state)
# print("Before transposing shape: ", state.shape)
# array_transposed = np.transpose(state, [1, 2, 0])
# print("array_transposed: ")
# print(array_transposed)
# print("array_transposed shape: ", array_transposed.shape)

# # batchsize = np.expand_dims(np.array(cur_stacked_images), axis=0)
# # print(state)
# # next_state = np.stack(cur_stacked_images, axis = 0)
# # print(next_state)