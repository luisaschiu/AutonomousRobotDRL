import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.python.keras import layers, initializers, models, optimizers, metrics, losses
from tensorflow.python.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
import cv2 as cv

# def build_model():
#         # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
#         # From Google article pseudocode line 2: Initialize action-value function Q with random weights
#         model = models.Sequential()
#         init = initializers.VarianceScaling(scale=2.0)
#         # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
#         model.add(layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=init))
#         model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=init))
#         model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=init))
#         model.add(layers.Flatten())
#         # Fully connected layer with 512 units, ReLU activation
#         model.add(layers.Dense(512, activation='relu', kernel_initializer=init))
#         # Output layer
#         model.add(layers.Dense(4, activation='linear', kernel_initializer=init))
#         return model

# Method with normalizing image
def build_model():
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        init = initializers.VarianceScaling(scale=2.0)
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (389, 398, 4), batch_size=32)
        # input_layer = Input(shape = (389, 398, 4))
        normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
        conv1 = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=init)(normalized_input)
        conv2 = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=init)(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=init)(conv2)
        flatten = Flatten()(conv3)
        # Fully connected layer with 512 units, ReLU activation
        dense1 = Dense(512, activation='relu', kernel_initializer=init)(flatten)
        # Output layer
        output_layer = Dense(4, activation='linear', kernel_initializer=init)(dense1)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

if __name__ == "__main__":
        # Initial parameters: create maze
        maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]])

        marker_filepath = "images/marker8.jpg"
        maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
        model = build_model()