import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.keras import initializers, models, optimizers, metrics
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
import cv2 as cv
import itertools

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

        self.init_exploration_rate = 1.0 # Exploration rate, also known as epsilon
        self.final_exploration_rate = 0.1
        self.final_exploration_frame = 1e6
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
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], self.agent_history_length), batch_size=self.minibatch_size)
        # input_layer = Input(shape = (389, 398, 4))
        normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
        conv1 = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(normalized_input)
        conv2 = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv2)
        flatten = Flatten()(conv3)
        # Fully connected layer with 512 units, ReLU activation
        dense1 = Dense(512, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.0))(flatten)
        # Output layer
        output_layer = Dense(4, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.0))(dense1)
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

    def remember(self, state, action, reward, next_state, game_over):
        self.replay_memory.append((state, action, reward, next_state, game_over))

    def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):
        """Use annealing schedule similar to: https://openai.com/blog/openai-baselines-dqn/ .

        Args:
            current_step (int): Number of entire steps agent experienced.
            terminal_eps (float): Final exploration rate arrived at terminal_frame_factor * self.final_exploration_frame.
            terminal_frame_factor (int): Final exploration frame, which is terminal_frame_factor * self.final_exploration_frame.

        Returns:
            eps (float): Calculated epsilon for ε-greedy at current_step.
        """
        terminal_eps_frame = self.final_exploration_frame * terminal_frame_factor
        # NOTE: self.replay_start_size is huge, about 10,000. May need to change this, or else we will want to explore for a long time.
        if current_step < self.replay_start_size:
            eps = self.init_exploration_rate
        # If the robot has taken enough steps before replaying old memories and updating the main model (greater than or equal to 
        # self.replay_start_size) and it is not at the last frame in which we want it to explore less.
        elif self.replay_start_size <= current_step and current_step < self.final_exploration_frame:
            eps = (self.final_exploration_rate - self.init_exploration_rate) / (self.final_exploration_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_exploration_rate
        # If the robot has taken enough steps as it gets closer to the final frames before it needs to be terminated to prevent over exploring
        elif self.final_exploration_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_exploration_rate) / (terminal_eps_frame - self.final_exploration_frame) * (current_step - self.final_exploration_frame) + self.final_exploration_rate
        else:
            # Right now, self.final_exploration_rate = 0.01. terminal_eps is 0.01. This means epsilon is very low, and 
            # there is a very low chance of exploration.
            eps = terminal_eps
        return eps
    
    def get_action(self, state, available_actions, expl_rate):
        #  This means that every value within the range [0, 1) has an equal probability of being chosen.
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            not_none_idx = [index for index, action in enumerate(available_actions) if action is not None]
            random_idx = random.choice(not_none_idx)
            return available_actions[random_idx]
        else:
            array =self.model.predict(state)
            # Copy array so we don't alter the original q-value array in case we want to look at it
            array_copy = array.copy()
            array_shape = array.shape
            best_action_idx = None
            while best_action_idx is None:
                max_idx = np.argmax(array_copy)
                col_idx = np.unravel_index(max_idx, array_shape)[1]
                if available_actions[col_idx] is not None:
                    best_action_idx = col_idx
                    break
                # If best_action is None, find the next largest q-value within the multidimensional array
                else:
                    array_copy.flat[max_idx] = np.iinfo(np.int32).min
            return available_actions[best_action_idx]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_main_model(self, state_batch, action_batch, reward_batch, next_state_batch, game_over_batch):
        """Update main q network by experience replay method.

        Args:
            state_batch (tf.float32): Batch of states.
            action_batch (tf.int32): Batch of actions.
            reward_batch (tf.float32): Batch of rewards.
            next_state_batch (tf.float32): Batch of next states.
            game_over_batch (tf.bool): Batch of game status.

        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """

        with tf.GradientTape() as tape:
            next_state_q = self.target_model.predict(next_state_batch)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch + self.discount_factor * next_state_max_q * (1.0 - tf.cast(game_over_batch, tf.float32))
            # tf.reduce_sum sums up all the Q-values for each sample in the batch.
            # tf.one_hot creates an encoding of the action batch with a depth of self.action_size.
            # main_q would theoretically yield a tensor vector of size (batch_size, action_size), which is (32, 4)
            main_q = tf.reduce_sum(self.model(state_batch) * tf.one_hot(action_batch, self.action_size, 1.0, 0.0), axis=1)
            loss = losses.Huber(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        # optimizer = optimizers.RMSProp(learning_rate= self.learning_rate),loss='mse') # From paper info, maybe misinterpreted?
        optimizer = optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-6)
        optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        metrics.Mean(name="loss").update_state(loss)
        metrics.Mean(name="Q_value").update_state(main_q)

        return loss
if __name__ == "__main__":
    # Initial parameters: create maze
    # Testing one run of the train_agent code:
    maze_array = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    marker_filepath = "images/marker8.jpg"
    maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
    network = DQN((389, 389))
    model = network.build_model()
    next_state_batch = []
    time_step = 0    
    init_state = maze.reset(time_step)
    state = network.preprocess_image(time_step, init_state)
    episode_score = 0


    # # Test logic for finding valid_indices of minibatches
    # # Initial parameters, changed to scale down and test/debug
    # replay_memory_capacity = 12
    # replay_memory = deque(maxlen=replay_memory_capacity)
    # minibatch_size = 10
    # agent_history_length = 4
    # # Create random batches for testing:
    # mem0 = ("state0", "action0", "reward0", "next_state0", False)
    # mem1 = ("state1", "action1", "reward1", "next_state1", False)
    # mem2 = ("state2", "action2", "reward2", "next_state2", False)
    # mem3 = ("state3", "action3", "reward3", "next_state3", False)
    # mem4 = ("state4", "action4", "reward4", "next_state4", False)
    # mem5 = ("state5", "action5", "reward5", "next_state5", True)
    # mem6 = ("state6", "action6", "reward6", "next_state6", False)
    # mem7 = ("state7", "action7", "reward7", "next_state7", False)
    # mem8 = ("state8", "action8", "reward8", "next_state8", False)
    # mem9 = ("state9", "action9", "reward9", "next_state9", False)
    # mem10 = ("state10", "action10", "reward10", "next_state10", False)
    # mem11 = ("state11", "action11", "reward11", "next_state11", False)
    # mem12 = ("state12", "action12", "reward12", "next_state12", False)
    # replay_memory.append(mem0)
    # replay_memory.append(mem1)
    # replay_memory.append(mem2)
    # replay_memory.append(mem3)
    # replay_memory.append(mem4)
    # replay_memory.append(mem5)
    # replay_memory.append(mem6)
    # replay_memory.append(mem7)
    # replay_memory.append(mem8)
    # replay_memory.append(mem9)
    # replay_memory.append(mem10)
    # replay_memory.append(mem11)
    # # replay_memory.append(mem12)
    # print(replay_memory)

    # # # Test slicing the deque to get the desired data within that interval
    # # sliced_deque = deque(itertools.islice(replay_memory, 0, 4))
    # # print("sliced: ", sliced_deque)

    # # Start of logic
    # indices_lst = []
    # cur_memory_size = len(replay_memory)
    # while len(indices_lst) < minibatch_size:
    #     # If replay memory is full and has hit it's maximum capacity, find a random index in the range: history length and memory_capacity
    #     if cur_memory_size == replay_memory_capacity:
    #         # NOTE: The np.random.randint is choosing from [low, high). I increased high by 1 to have it be considered.
    #         # NOTE: We index by 0, so should I lower the low value by 1?
    #         index = np.random.randint(low=agent_history_length, high=(replay_memory_capacity+1), dtype=np.int32)
    #     else:
    #     # If replay memory isn't full yet, sample from existing replay memory
    #     # NOTE: The np.random.randint is choosing from [low, high). I increased high by 1 to have it be considered.
    #         index = np.random.randint(low=agent_history_length, high=(cur_memory_size+1), dtype=np.int32)
    #     # If any cases are terminal, disregard and keep looking for a new random index to add onto the list
    #     print("index: ", index)
    #     sliced_deque = deque(itertools.islice(replay_memory, (index-agent_history_length), (index)))
    #     terminal_flag = False
    #     for item in sliced_deque:
    #         if item[4] == True:
    #             terminal_flag = True
    #             print(item[0], " won't work")
    #             break
    #         print(item[0], "works")
    #     if terminal_flag == False:
    #         indices_lst.append((index-1))
    # print(indices_lst)
    # state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = [], [], [], [], []
    # for index in indices_lst:
    #     (state, action, reward, next_state, game_over) = replay_memory[index]
    #     state_batch.append(state)
    #     game_over_batch.append(game_over)
    # print(state_batch)
    # print(game_over_batch)


    # # Test inputting varying batch sizes in NN model
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # marker_filepath = "images/marker8.jpg"
    # maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
    # network = DQN((389, 389))
    # model = network.build_model()
    # next_state_batch = []
    # time_step = 0    
    # init_state = maze.reset(time_step)
    # state = network.preprocess_image(time_step, init_state)
    # episode_score = 0

    # # Test inputting a batch size of 1
    # # q_val = model.predict(state)
    # # print(q_val)

    # # Start of while loop in train_agent:
    # time_step += 1
    # action = "DOWN"
    # (next_state_img, reward, game_over) = maze.take_action(action, time_step)
    # episode_score += reward
    # next_state = network.preprocess_image(time_step, next_state_img)
    # network.remember(state, action, reward, next_state, game_over)
    # # print()
    # # print("First next_state: ")
    # # print(next_state)
    # # print(next_state.shape)
    # next_state_batch.append(tf.constant(next_state, tf.float32))
    # # print(network.replay_memory)
    # # print(len(network.replay_memory))
    # state = next_state
    # time_step += 1
    # action = "RIGHT"
    # (next_state_img, reward, game_over) = maze.take_action(action, time_step)
    # episode_score += reward
    # next_state = network.preprocess_image(time_step, next_state_img)
    # network.remember(state, action, reward, next_state, game_over)
    # # print()
    # # print("Second next_state: ")
    # # print(next_state)
    # # print(next_state.shape)
    # next_state_batch.append(tf.constant(next_state, tf.float32))
    # concatenated_tensor = next_state_batch[0]  # Initialize with the first tensor
    # for i in range(1, len(next_state_batch)):
    #     concatenated_tensor = tf.concat([concatenated_tensor, next_state_batch[i]], axis=0)
    # # Print the concatenated tensor
    # print("Concatenated tensor along batch size dimension:")
    # print(concatenated_tensor)

    # # Test inputting a batch size of larger than 1
    # tester =  model.predict(concatenated_tensor)
    # print(tester)
    # tester =  model.predict(state)


    # # Test output shape of batches that are not state or next_state
    # game_over_batch = []
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_batch.append(tf.constant(False, tf.bool))
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_tensor = tf.stack(game_over_batch, axis=0)
    # print(game_over_tensor)


    # # Test finding highest max_val in a multi-dimensional array and choosing next best action(max q value) if the "best_action" is unavailable.
    # available_actions = [None, "DOWN", None, "RIGHT"]
    # arr = np.array([[20, 3, 0, 4],
    #                 [3, -5, 60, 2],
    #                 [3000, -3300, 3, -3]])
    # # Copy array
    # array_copy = arr.copy()
    # array_shape = arr.shape
    # valid_idx = None
    # while valid_idx is None:
    #     max_idx = np.argmax(array_copy)
    #     col_idx = np.unravel_index(max_idx, array_shape)[1]
    #     if available_actions[col_idx] is not None:
    #         valid_idx = col_idx
    #         break
    #     else:
    #         array_copy.flat[max_idx] = np.iinfo(np.int32).min  # Set original max value to a very low integer
    # print(available_actions[col_idx])


    # # Find index of max value and find next highest:
    # max_index = np.argmax(arr)
    # # Copy array and set the maximum value to a very low number
    # arr_copy = arr.copy()
    # arr_copy.flat[max_index] = np.iinfo(np.int32).min  # Set original max value to a very low integer
    # # Find index of next highest value
    # next_highest_index_flat = np.argmax(arr_copy)
    # # Convert flat index to index in the original array
    # next_highest_index = np.unravel_index(next_highest_index_flat, arr.shape)
    # # print("Max index value:", max_index)
    # print("Index of maximum value:", np.unravel_index(max_index, arr.shape))
    # # print("Next max index value:", next_highest_index_flat)
    # print("Index of next highest value:", next_highest_index)



    # Test exploring and choosing a random action in available_actions list:
    # available_actions = [None, "DOWN", None, "RIGHT"]
    # not_none_idx = [index for index, action in enumerate(available_actions) if action is not None]
    # For exploring:
    # print(not_none_idx)
    # random_idx = random.choice(not_none_idx)
    # print(available_actions[random_idx])


    # An idea to vary batchsize:
    # # Define a placeholder for the batch size 
    # batch_size = None
    # # Create a placeholder tensor with shape (batch_size, height, width, channels)
    # input_tensor = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 4))