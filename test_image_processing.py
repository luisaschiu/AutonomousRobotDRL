import numpy as np
from collections import deque
import tensorflow as tf
import cv2 as cv
import glob
import os


''' Test resizing images'''
image = cv.imread('test_resize_img.jpg')
print(image.shape)
resized = cv.resize(image, (120, 120))
cv.imshow('img', resized)
cv.waitKey(0)

''' Test stacking images'''
# Test stacking images
# arr1 = np.array(
#         [[1, 1, 1, 1],
#          [1, 1, 1, 1],
#          [1, 1, 1, 1],
#          [1, 1, 1, 1]])
# arr2 = np.array(
#         [[2, 2, 2, 2],
#          [2, 2, 2, 2],
#          [2, 2, 2, 2],
#          [2, 2, 2, 2]])
# arr3 = np.array(
#         [[3, 3, 3, 3],
#          [3, 3, 3, 3],
#          [3, 3, 3, 3],
#          [3, 3, 3, 3]])
# arr4 = np.array(
#         [[4, 4, 4, 4],
#          [4, 4, 4, 4],
#          [4, 4, 4, 4],
#          [4, 4, 4, 4]])
# arr5 = np.array(
#         [[5, 5, 5, 5],
#          [5, 5, 5, 5],
#          [5, 5, 5, 5],
#          [5, 5, 5, 5]])
# # Testing expected output using np.stack method:
# # NOTE: I am not going with this approach, since using deque would be easier to handle keeping a maximum of 4 sequential time frames.
# # This is to show the expected result for the methods using deque.
# print("arr1 shape:", str(arr1.shape))
# stacked_arr = np.stack((arr1, arr2, arr3), axis=-1, dtype=np.float32)
# print("Stacked array/Expected Result: ")
# print( stacked_arr)
# print("Stacked array/Expected Result shape: ")
# print(stacked_arr.shape)

# # Using deque: stack images by appending to deque object type, transposes to expected result shape
# # NOTE: For now, I am going with the tensorflow object type since it makes more sense to me, inputting a tensor into the neural network

# # Method 1: Convert to tensorflow object type
# replay_memory = deque(maxlen=3)
# cur_stacked_images = deque(maxlen = 4)
# cur_stacked_images.append(arr1)
# cur_stacked_images.append(arr2)
# cur_stacked_images.append(arr3)
# cur_stacked_images.append(arr4)
# # Assign to state variable, similar to what we will do in class_DQN
# state = cur_stacked_images
# print("Before converting to tensor: ")
# print(state)
# tensor = tf.constant(cur_stacked_images, tf.float32)
# print("After converting to tensor: ")
# print(tensor)
# tensor_transposed = tf.transpose(tensor, [1, 2, 0])
# print("tensor_ transposed: ")
# print(tensor_transposed)
# print("Add batchsize dimension: ")
# batchsize = tf.expand_dims(tensor_transposed, axis=0)
# replay_memory.append(batchsize)
# print(batchsize)

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


''' Test limit generated images'''
# Test counting number of files in directory
# count = 0
# robot_images = [cv.imread(image) for image in glob.glob("test_robot_steps/*.jpg")]
# for image in robot_images:
#     # cv.imshow('img', image)
#     # cv.waitKey(1000)
#     count += 1
# print(count)

# # Test copying files over
# # Set the current working directory
# os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
# # Set the source and the destination folders
# src = os.getcwd() + "\\test_robot_steps"
# dst = os.getcwd() + "\\test_images"
# print(src)
# # Copy file
# os.system('copy ' + src+'\\0.jpg ' + dst + '\\0.jpg') # NOTE: can rename file when copying over to dst if desired.

# # Test deleting a file
# os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
# dst = os.getcwd() + "\\test_images"
# os.remove(dst +'\\1.jpg')

# The following code copies images from a src folder into a dst folder, as if it is generating an image every time step.
# The dst folder saves each time step and gets rid of the oldest one once it saves 3 time steps.
# cur_num = 0
# count = 0
# os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
# src = os.getcwd() + "\\test_robot_steps"
# dst = os.getcwd() + "\\test_images"
# while True:
#     input_char = input("Click 'y' to generate an image, 'e' to quit: ")
#     if input_char == 'y':
#         os.system('copy ' + src +'\\' + str(cur_num) + '.jpg ' + dst + '\\' + str(cur_num) + '.jpg')
#         images = [cv.imread(image) for image in glob.glob("test_images/*.jpg")]
#         for image in images:
#             count += 1
#         if count > 3:
#             os.remove(dst + '\\' + str(cur_num-3) + '.jpg')
#         cur_num += 1
#         count = 0
#     elif input_char == 'e':
#         exit()
# # Possible TODO: account for edge case for cur_num exceeding number of images in src directory

