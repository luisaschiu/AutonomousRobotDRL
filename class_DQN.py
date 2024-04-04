import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze, deleteGifs
from tensorflow.keras import initializers, models, optimizers, metrics, losses
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input, Rescaling
import cv2 as cv
import itertools
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import threading
import math
import pickle
import os

class DQN:
    def __init__(self, state_size, maze_size):
        # State size is the image size
        self.state_size = state_size
        self.action_size = 4
        # From Google article pseudocode line 1: Initialize replay memory D to capacity N
        self.replay_memory_capacity=10000000
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.replay_start_size = maze_size**3 # nrows^3
        self.discount_factor = 0.99 # Also known as gamma
        self.init_exploration_rate = 1.0 # Exploration rate, also known as epsilon
        self.final_exploration_rate = 0.1
        # self.final_exploration_frame = 12  # This performed better than the past
        self.final_exploration_frame = maze_size*250 # Josh uses: (nrows^3*5)
        self.learning_rate = 0.001
        self.minibatch_size = 32
        self.max_steps_per_episode = maze_size*5 # nrows^2
        self.win_history = []
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.model = self.build_model()
        self.target_model = models.clone_model(self.model)
        self.update_target_network_freq = maze_size*6 
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        self.target_model.set_weights(self.model.get_weights())
        # optimizer = optimizers.RMSProp(learning_rate= self.learning_rate),loss='mse') # From paper info, maybe misinterpreted?
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-6)
        self.loss_metric = metrics.Mean(name="loss")
        self.Q_value_metric = metrics.Mean(name="Q_value")
        self.actions_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.episode_rewards_lst = []
        self.loss_lst = []
        self.total_step_loss_lst = []
        self.expl_rate_lst = []

    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], self.agent_history_length), batch_size=self.minibatch_size)
        # input_layer = Input(shape = (389, 398, 4))
        # normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
        normalized_input = Rescaling(scale=1.0/255.0)(input_layer)
        # print(normalized_input)
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

    def get_action(self, state, available_actions, expl_rate):
        # #  This means that every value within the range [0, 1) has an equal probability of being chosen.
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            # Filter available actions
            valid_actions = [action for action, is_available in zip(self.actions_list, available_actions) if is_available]
            return random.choice(valid_actions)
        else:
            array=self.model.predict(state)
            # Copy array so we don't alter the original q-value array in case we want to look at it
            # print(array)
            masked_qval_array = np.where(np.array(available_actions) == 1, array, float('-inf'))
            # print(masked_qval_array)
            max_val_index = np.argmax(np.max(masked_qval_array, axis=0))
            return self.actions_list[max_val_index]
        
    def remember(self, state, action, reward, next_state, game_over, next_state_available_action):
        self.replay_memory.append((state, action, reward, next_state, game_over, next_state_available_action))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # NOTE: get_eps function taken from Atari game. Used to calculate epsilon value for epsilon-greedy policy based on an annealing schedule.
    # def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):
    #     """Use annealing schedule similar to: https://openai.com/blog/openai-baselines-dqn/ .

    #     Args:
    #         current_step (int): Number of entire steps agent experienced.
    #         terminal_eps (float): Final exploration rate arrived at terminal_frame_factor * self.final_exploration_frame.
    #         terminal_frame_factor (int): Final exploration frame, which is terminal_frame_factor * self.final_exploration_frame.

    #     Returns:
    #         eps (float): Calculated epsilon for ε-greedy at current_step.
    #     """
    #     terminal_eps_frame = self.final_exploration_frame * terminal_frame_factor
    #     # NOTE: self.replay_start_size is huge, about 10,000. May need to change this, or else we will want to explore for a long time.
    #     if current_step < self.replay_start_size:
    #         # print("In if statement")
    #         eps = self.init_exploration_rate
    #     # If the robot has taken enough steps before replaying old memories and updating the main model (greater than or equal to 
    #     # self.replay_start_size) and it is not at the last frame in which we want it to explore less.
    #     elif self.replay_start_size <= current_step and current_step < self.final_exploration_frame:
    #         # print("In 1st elif statement")
    #         eps = (self.final_exploration_rate - self.init_exploration_rate) / (self.final_exploration_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_exploration_rate
    #     # If the robot has taken enough steps as it gets closer to the final frames before it needs to be terminated to prevent over exploring
    #     elif self.final_exploration_frame <= current_step and current_step < terminal_eps_frame:
    #         # print("In 2nd elif statement")
    #         eps = (terminal_eps - self.final_exploration_rate) / (terminal_eps_frame - self.final_exploration_frame) * (current_step - self.final_exploration_frame) + self.final_exploration_rate
    #     else:
    #         # Right now, self.final_exploration_rate = 0.01. terminal_eps is 0.01. This means epsilon is very low, and 
    #         # there is a very low chance of exploration.
    #         # print("In else statement")
    #         eps = terminal_eps
    #     return eps
    
    def get_eps(self, steps):
        # return math.exp(-0.008*steps) # Good fo 4x4 maze
        return math.exp(-0.004*steps)

    
    @tf.function
    def update_main_model(self, state_batch, action_batch, reward_batch, next_state_batch, game_over_batch, next_state_available_actions_batch):
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
            next_state_q = self.target_model(next_state_batch)
            # print("next_state_q")
            # print(next_state_q)
            # tf.print(next_state_q)
            # Replace unavailable actions with -infinity
            masked_q_tensor = tf.where(next_state_available_actions_batch == 1, next_state_q, tf.constant(float('-inf'), shape=next_state_q.shape))
            # print(masked_qval_tensor)
            # Find largest q value within masked q tensor
            next_state_max_q = tf.math.reduce_max(masked_q_tensor, axis=1)
            # print(largest_values)
            # next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            # print("next_state_max_q")
            # print(next_state_max_q)
            # tf.print(next_state_max_q)
            # Computes the expected Q-value using the Bellman equation.
            expected_q = reward_batch + self.discount_factor * next_state_max_q * (1.0 - tf.cast(game_over_batch, tf.float32))
            # tf.reduce_sum sums up all the Q-values for each sample in the batch.
            # tf.one_hot creates an encoding of the action batch with a depth of self.action_size.
            # main_q would theoretically yield a tensor vector of size (batch_size, action_size), which is (32, 4)
            unique_actions = tf.constant(["UP", "DOWN", "LEFT", "RIGHT"])  # Get unique actions as a TensorFlow constant
            action_indices = tf.argmax(tf.cast(tf.equal(unique_actions[:, tf.newaxis], action_batch), tf.int32), axis=0)
            # print(action_indices)
            action_one_hot = tf.one_hot(action_indices, depth=self.action_size, on_value=1.0, off_value=0.0)
            main_q = tf.reduce_sum(self.model(state_batch) * action_one_hot, axis=1)
            # Output loss val tensor shape: (32,)
            main_q_dim = tf.expand_dims(main_q, axis = 1)
            expected_q_dim = tf.expand_dims(expected_q, axis = 1)
            # print(main_q_dim)
            # print(expected_q_dim)
            loss = losses.Huber(reduction=losses.Reduction.NONE)
            loss_val = loss(tf.stop_gradient(expected_q_dim), main_q_dim)
            # print(loss_val)

        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        self.loss_metric.update_state(loss_val)
        self.Q_value_metric.update_state(main_q)

        avg_loss = tf.math.reduce_mean(loss_val)
        return avg_loss

    # Generate batches of random memories pulled from self.replay_memory
    def generate_minibatch_samples(self):
        # Generate list of random indices
        indices_lst = []
        cur_memory_size = len(self.replay_memory)
        while len(indices_lst) < self.minibatch_size:
            # If replay memory is full and has hit it's maximum capacity, find a random index in the range: history length and memory_capacity
            if cur_memory_size == self.replay_memory_capacity:
                # The np.random.randint is choosing from [low, high). I increased high by 1 to have it be considered.
                index = np.random.randint(low=self.agent_history_length, high=(self.replay_memory_capacity+1), dtype=np.int32)
            else:
            # If replay memory isn't full yet, sample from existing replay memory
            # The np.random.randint is choosing from [low, high). I increased high by 1 to have it be considered.
                index = np.random.randint(low=self.agent_history_length, high=(cur_memory_size+1), dtype=np.int32)
            # If any cases are terminal, disregard and keep looking for a new random index to add onto the list
            sliced_deque = deque(itertools.islice(self.replay_memory, (index-self.agent_history_length), (index)))
            terminal_flag = False
            for item in sliced_deque:
                if item[4] == True:
                    terminal_flag = True
                    break
            if terminal_flag == False:
                # Since slicing the deque doesn't consider the last index, I have to offset the index by 1.
                # Slice notation [start:stop] extracts elements from the index start up to, but not including, the index stop.
                indices_lst.append(index-1)
        # If going through all of those for loops are too computationally intensive, try this code from chatgpt:
        # # Extract data from self.replay_memory based on indices_lst
        # replay_data = [self.replay_memory[index] for index in indices_lst]
        # # Separate the data into individual lists
        # state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = zip(*replay_data)
        # # Convert lists to tensors
        # action_batch = tf.stack([tf.constant(action, dtype=tf.int32) for action in action_batch])
        # reward_batch = tf.stack([tf.constant(reward, dtype=tf.float32) for reward in reward_batch])
        # game_over_batch = tf.stack([tf.constant(game_over, dtype=tf.bool) for game_over in game_over_batch])

        state_batch, action_batch, reward_batch, next_state_batch, game_over_batch, next_state_available_actions_batch = [], [], [], [], [], []
        for index in indices_lst:
            (state, action, reward, next_state, game_over, next_state_available_actions) = self.replay_memory[index]
            state_batch.append(tf.constant(state, tf.float32))
            action_batch.append(tf.constant(action, tf.string))
            reward_batch.append(tf.constant(reward, tf.float32))
            next_state_batch.append(tf.constant(next_state, tf.float32))
            game_over_batch.append(tf.constant(game_over, tf.bool))
            next_state_available_actions_batch.append(tf.constant(next_state_available_actions, tf.int32))
        # Organize the batch_size to have proper dimensions for state_batch and next_state_batch:
        # Initialize with the first tensor
        concatenated_state_tensor = state_batch[0] 
        for i in range(1, len(next_state_batch)):
            concatenated_state_tensor = tf.concat([concatenated_state_tensor, state_batch[i]], axis=0)
        # Repeat for next_state_batch. Initialize with the first tensor.
        concatenated_next_state_tensor = next_state_batch[0]
        for i in range(1, len(next_state_batch)):
            concatenated_next_state_tensor = tf.concat([concatenated_next_state_tensor, next_state_batch[i]], axis=0)
        # NOTE: action_batch, reward_batch, and game_over_batch will all have a tensor flow shape of: shape=(4,). Found through testing.
        return concatenated_state_tensor, tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), concatenated_next_state_tensor, tf.stack(game_over_batch, axis=0), tf.stack(next_state_available_actions_batch, axis=0)

    def preprocess_image(self, time_step, new_image):
        # Get rid of the 3 color channels, convert to grayscale
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        new_image = cv.resize(new_image, (120, 120))
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

    def save_to_csv(self, data, file_path, headers=None):
        """
        Save data to a CSV file.

        Parameters:
            data: Data to be saved.
            file_path (str): Path to the CSV file.
            headers (list, optional): List of header names. Defaults to None.
        """
        mode = 'w' if headers else 'a'  # Use 'w' mode initially if headers are provided, otherwise use 'a' mode
        # print("mode: ", mode)
        with open(file_path, mode, newline="") as file:
            writer = csv.writer(file)
            if headers:
                writer.writerow(headers)
            writer.writerow(data)

    def save_plots(self):
        data_folder_path = 'data_plots'
        os.makedirs(data_folder_path, exist_ok = True)
        rwd_fig = plt.figure()
        # plt.clf()
        # plt.plot([i for i in range(0, len(rewards_lst))], rewards_lst, color='blue', linestyle='-', label='Lines')
        plt.plot([i for i in range(0, len(self.episode_rewards_lst))], self.episode_rewards_lst)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(os.path.join(data_folder_path + '/rewards.png'))
        pickle.dump(rwd_fig, open(data_folder_path + '/rewards.pickle', 'wb'))
        plt.clf()

        # plt.plot(total_step_loss_lst, loss_lst, color='blue', linestyle='-', label='Lines')
        loss_fig = plt.figure()
        plt.plot(self.total_step_loss_lst, self.loss_lst)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(data_folder_path + '/loss.png'))
        pickle.dump(loss_fig, open(data_folder_path + '/loss.pickle', 'wb'))
        plt.clf()

        # plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst, color='blue', linestyle='-', label='Lines')
        expl_fig = plt.figure()
        plt.plot([i for i in range(0, len(self.expl_rate_lst))], self.expl_rate_lst)
        plt.xlabel('Steps')
        plt.ylabel('Expl_rate')
        plt.savefig(os.path.join(data_folder_path + '/expl_rate.png'))
        pickle.dump(expl_fig, open(data_folder_path + '/expl_rate.pickle', 'wb'))
        plt.clf()
    

    # def animate(self):
    #     data = pd.read_csv('data.csv')
    #     x = data['Episode']
    #     y = data['Reward']
    #     plt.cla()
    #     plt.plot(x, y)

    # def plot_thread(self):
    #     fig = plt.figure(2)  # Create a new figure
    #     ani = FuncAnimation(fig, self.animate())  # Create the animation
    #     plt.show()  # Show the plot and animation

    # def save_weights(self):
    #     model_json = self.model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     self.model.save_weights("model.h5")
    #     print("Saved model to disk")
    
    # def load_weights(self):
    #     json_file = open('model.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     # loaded_model = models.model_from_json(loaded_model_json, safe_mode = False)
    #     loaded_model = models.model_from_json(loaded_model_json)  # Load model architecture from JSON
    #     loaded_model.load_weights("model.h5")  # Load weights separately
    #     print("Loaded model from disk")

    def train_agent_static(self, maze: Maze, num_episodes = 200):
        loss = 0
        total_step = 0
        deleteGifs()
        for episode in range(num_episodes):
            self.cur_stacked_images.clear()
            episode_step = 0
            episode_score = 0.0
            game_over = False
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1).
            init_state = maze.reset(episode_step)
            state = self.preprocess_image(episode_step, init_state)
            while not game_over:
                # From Google article pseudocode line 5: With probability epsilon select a random action a_t
                expl_rate = self.get_eps(total_step)
                self.expl_rate_lst.append(expl_rate)
                available_actions = maze.get_available_actions()
                action = self.get_action(state, available_actions, expl_rate)
                total_step += 1
                episode_step += 1
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
                episode_score += reward
                next_state_available_actions = maze.get_available_actions()
                # next_state_available_actions_filtered = [0 if x is None else x for x in next_state_available_actions]
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = self.preprocess_image(episode_step, next_state_img)
                # From Google article pseudocode line 8: Store transition/experience in D(replay memory)
                self.remember(state, action, reward, next_state, game_over, next_state_available_actions)
                state = next_state
                if (total_step % self.agent_history_length == 0) and (total_step > self.replay_start_size):
                    print("Generating minibatch and updating main model")
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch = self.generate_minibatch_samples()
                    # print("next_state_available_actions_batch")
                    # print(next_state_available_actions_batch)
                    loss = self.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch)
                    self.total_step_loss_lst.append(total_step)
                    self.loss_lst.append(loss.numpy())
                    # print('Loss: ' + str(loss.numpy()))
                if episode_step == self.max_steps_per_episode:
                    game_over = True
                # From Google article pseudocode line 12: Every C steps reset Q^hat = Q
                if ((total_step % self.update_target_network_freq == 0) and (total_step > self.replay_start_size)):
                    self.update_target_model()
                # From Google article pseudocode line 10: if episode terminates at step j+1
                if game_over:
                    self.episode_rewards_lst.append(episode_score)
                    print('Game Over.')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
                    maze.produce_video(str(episode), 'training_episode_videos')
                    # break
            if episode == 0:
                self.save_to_csv([episode, episode_score, episode_step, expl_rate], "training_data.csv", ["Episode", "Reward", "Steps", "Expl Rate"])
            else:
                self.save_to_csv([episode, episode_score, episode_step, expl_rate], "training_data.csv", None)
        self.model.save_weights("model_weights.h5")
        self.save_plots()
            # plot_thread = threading.Thread(target=self.plot_thread, daemon=True)
            # plot_thread.start()
                # print("total steps: ", total_step)
                    
                # if game_over == 'win':
                #     self.win_history.append(1)
                #     print('win') #TODO: Finish this print statement to provide more information
                #     break
                # elif game_over == 'lose':
                #     self.win_history.append(0)
                #     print('lose')
                #     break
                # If episode does not terminate... continue onto last lines of pseudocode
                # From Google article pseudocode line 11: Perform a gradient descent step (done in update_main_model)
    
    def train_agent_dynamic(self, maze_lst, num_episodes = 200):
        loss = 0
        total_step = 0
        deleteGifs()
        for episode in range(num_episodes):
            self.cur_stacked_images.clear()
            episode_step = 0
            episode_score = 0.0
            game_over = False
            maze = random.choice(maze_lst)
            # maze.show()
            index_of_maze = maze_lst.index(maze)
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1).
            init_state = maze.reset(episode_step)
            state = self.preprocess_image(episode_step, init_state)
            while not game_over:
                # From Google article pseudocode line 5: With probability epsilon select a random action a_t
                expl_rate = self.get_eps(total_step)
                self.expl_rate_lst.append(expl_rate)
                available_actions = maze.get_available_actions()
                print(available_actions)
                action = self.get_action(state, available_actions, expl_rate)
                print(action)
                total_step += 1
                episode_step += 1
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
                episode_score += reward
                next_state_available_actions = maze.get_available_actions()
                # next_state_available_actions_filtered = [0 if x is None else x for x in next_state_available_actions]
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = self.preprocess_image(episode_step, next_state_img)
                # From Google article pseudocode line 8: Store transition/experience in D(replay memory)
                self.remember(state, action, reward, next_state, game_over, next_state_available_actions)
                state = next_state
                if (total_step % self.agent_history_length == 0) and (total_step > self.replay_start_size):
                    print("Generating minibatch and updating main model")
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch = self.generate_minibatch_samples()
                    # print("next_state_available_actions_batch")
                    # print(next_state_available_actions_batch)
                    loss = self.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch)
                    self.total_step_loss_lst.append(total_step)
                    self.loss_lst.append(loss.numpy())
                    # print('Loss: ' + str(loss.numpy()))
                if episode_step == self.max_steps_per_episode:
                    game_over = True
                # From Google article pseudocode line 12: Every C steps reset Q^hat = Q
                if ((total_step % self.update_target_network_freq == 0) and (total_step > self.replay_start_size)):
                    self.update_target_model()
                # From Google article pseudocode line 10: if episode terminates at step j+1
                if game_over:
                    self.episode_rewards_lst.append(episode_score)
                    print('Game Over.')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
                    maze.produce_video(str(episode), 'training_episode_videos')
                    # break
            if episode == 0:
                self.save_to_csv([episode, episode_score, episode_step, expl_rate, index_of_maze], "training_data.csv", ["Episode", "Reward", "Steps", "Expl Rate", "Maze Num"])
            else:
                self.save_to_csv([episode, episode_score, episode_step, expl_rate, index_of_maze], "training_data.csv", None)
        self.model.save_weights("model_weights.h5")
        self.save_plots()
            # plot_thread = threading.Thread(target=self.plot_thread, daemon=True)
            # plot_thread.start()
                # print("total steps: ", total_step)
                    
                # if game_over == 'win':
                #     self.win_history.append(1)
                #     print('win') #TODO: Finish this print statement to provide more information
                #     break
                # elif game_over == 'lose':
                #     self.win_history.append(0)
                #     print('lose')
                #     break
                # If episode does not terminate... continue onto last lines of pseudocode
                # From Google article pseudocode line 11: Perform a gradient descent step (done in update_main_model)

    def play_game_static(self, maze:Maze, num_episodes, load_weight_dir=None):
        print("Playing game...")
        if load_weight_dir is not None:
            self.model.load_weights(load_weight_dir)
        total_step = 0
        maze.deleteGifs()
        for episode in range(num_episodes):
            status = None
            self.cur_stacked_images.clear()
            episode_step = 0
            episode_score = 0.0
            game_over = False
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1). NOTE: We do not downsize our image in preprocessing just yet.
            init_state = maze.reset(episode_step)
            state = self.preprocess_image(episode_step, init_state)
            while not game_over:
                available_actions = maze.get_available_actions()
                action = self.get_action(state, available_actions, 0.01)
                total_step += 1
                episode_step += 1
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
                episode_score += reward
                # next_state_available_actions_filtered = [0 if x is None else x for x in next_state_available_actions]
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = self.preprocess_image(episode_step, next_state_img)
                state = next_state
                if episode_step == self.max_steps_per_episode:
                    status = 0
                    game_over = True
                if game_over:
                    if status == 0:
                        print('Game Over. LOSE!')
                    else:
                        status = 1
                        print('Game Over. WIN!')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
                    # save data to .csv file
                    if episode == 0:
                        self.save_to_csv([episode, episode_score, episode_step, status], "gameplay_data.csv", ["Episode", "Reward", "Steps", "Game Status"])
                    else:
                        self.save_to_csv([episode, episode_score, episode_step, status], "gameplay_data.csv", None)
                    maze.produce_video(str(episode), 'gameplay_episode_videos')

    def play_game_dynamic(self, maze_lst, num_episodes, load_weight_dir=None):
        print("Playing game...")
        if load_weight_dir is not None:
            self.model.load_weights(load_weight_dir)
        total_step = 0
        deleteGifs()
        for episode in range(num_episodes):
            status = None
            self.cur_stacked_images.clear()
            episode_step = 0
            episode_score = 0.0
            game_over = False
            maze = random.choice(maze_lst)
            index_of_maze = maze_lst.index(maze)
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1). NOTE: We do not downsize our image in preprocessing just yet.
            init_state = maze.reset(episode_step)
            state = self.preprocess_image(episode_step, init_state)
            while not game_over:
                available_actions = maze.get_available_actions()
                action = self.get_action(state, available_actions, 0.01)
                total_step += 1
                episode_step += 1
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
                episode_score += reward
                # next_state_available_actions_filtered = [0 if x is None else x for x in next_state_available_actions]
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = self.preprocess_image(episode_step, next_state_img)
                state = next_state
                if episode_step == self.max_steps_per_episode:
                    status = 0
                    game_over = True
                if game_over:
                    if status == 0:
                        print('Game Over. LOSE!')
                    else:
                        status = 1
                        print('Game Over. WIN!')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
                    # save data to .csv file
                    if episode == 0:
                        self.save_to_csv([episode, episode_score, episode_step, status, index_of_maze], "gameplay_data.csv", ["Episode", "Reward", "Steps", "Game Status", "Maze Num"])
                    else:
                        self.save_to_csv([episode, episode_score, episode_step, status, index_of_maze], "gameplay_data.csv", None)
                    maze.produce_video(str(episode), 'gameplay_episode_videos')

def show_pickle_figure(pickle_file_path: str):
    figure = pickle.load(open(pickle_file_path, 'rb'))
    figure.axes[0].lines[0].get_data()
    figure.show()
    plt.show(block=True)

def save_pickle_to_csv(save_csv_file_path: str, pickle_file_path:str):
    figure = pickle.load(open(pickle_file_path, 'rb'))
    data = figure.axes[0].lines[0].get_data()
    lst = list(zip(data[0], data[1]))
    with open(save_csv_file_path+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x-values', 'y-values'])
        writer.writerows(lst)