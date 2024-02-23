import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.python.keras import layers, initializers, models, optimizers, metrics, losses
from tensorflow.python.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
from PIL import Image
import cv2 as cv

class DQN:
    def __init__(self, state_size):
        # State size is the image size
        self.state_size = state_size
        self.action_size = 4
        # From Google article pseudocode line 1: Initialize replay memory D to capacity N
        self.replay_memory_capacity=10000000
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.minibatch_size = 32
        self.max_timesteps = 20 # TODO: Chosen arbitrarily right now, make sure you change this as needed
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network_freq = 1000
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        # NOTE: Bottom line of code might be redundant, unless I include the feature where I load existing trained weights from a given file
        self.target_model.set_weights(self.model.get_weights())

    # def build_model():
    #     # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
    #     # From Google article pseudocode line 2: Initialize action-value function Q with random weights
    #     model = models.Sequential()
    #     init = initializers.VarianceScaling(scale=2.0)
    #     # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
    #     model.add(layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=init))
    #     model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=init))
    #     model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=init))
    #     model.add(layers.Flatten())
    #     # Fully connected layer with 512 units, ReLU activation
    #     model.add(layers.Dense(512, activation='relu', kernel_initializer=init))
    #     # Output layer
    #     model.add(layers.Dense(4, activation='linear', kernel_initializer=init))
    #     return model

    # Method with normalizing image
    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        init = initializers.VarianceScaling(scale=2.0)
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], 4), batch_size=self.minibatch_size)
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

    def preprocess_image(self, time_step, new_image):
        # Get rid of the 3 color channels, convert to grayscale
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        # If it is the start of the game (time_step = 0), append the start configuration 4 times as initial input to the neural network model.
        if time_step == 0:
            self.cur_stacked_images.append(new_image)
            self.cur_stacked_images.append(new_image)
            self.cur_stacked_images.append(new_image)
            self.cur_stacked_images.append(new_image)
            tensor = tf.constant(self.cur_stacked_images, tf.float32) # Convert to tensor type, make sure all values are of datatype float32
            tensor_transposed = tf.transpose(tensor, [1, 2, 0]) # Change tensor to desired shape (img_height, img_width, self.agent_history_length)
            tensor_batch = tf.expand_dims(tensor_transposed, axis=0)  # Adding batch dimension
        else:
            self.cur_stacked_images.append(new_image)
            tensor = tf.constant(self.cur_stacked_images, tf.float32) # Convert to tensor type, make sure all values are of datatype float32
            tensor_transposed = tf.transpose(tensor, [1, 2, 0]) # Change tensor to desired shape (img_height, img_width, self.agent_history_length)
            tensor_batch = tf.expand_dims(tensor_transposed, axis=0)  # Adding batch dimension
        return tensor_batch

if __name__ == "__main__":
    # Initial parameters: create maze
    maze_array = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])

    marker_filepath = "images/marker8.jpg"
    maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
    network = DQN((389, 389))
    model = network.build_model()
    time_step = 0
    init_state = maze.reset(time_step)
    state = network.preprocess_image(time_step, init_state)
    # model.

    # An idea to vary batchsize:
    # # Define a placeholder for the batch size 
    # batch_size = None

    # # Create a placeholder tensor with shape (batch_size, height, width, channels)
    # input_tensor = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 4))