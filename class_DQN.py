import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.keras import initializers, models, optimizers, metrics, losses
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
        self.replay_start_size = 1e4
        self.discount_factor = 0.99 # Also known as gamma
        self.init_exploration_rate = 1.0 # Exploration rate, also known as epsilon
        self.final_exploration_rate = 0.1
        self.final_exploration_frame = 1e6
        self.learning_rate = 0.001
        self.minibatch_size = 32
        self.max_timesteps = 20 # TODO: Chosen arbitrarily right now, make sure you change this as needed
        self.win_history = []
        self.model = self.build_model()
        
        self.target_model = self.build_model()
        self.update_target_network_freq = 1000
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        # NOTE: Bottom line of code might be redundant, unless I include the feature where I load existing trained weights from a given file
        self.target_model.set_weights(self.model.get_weights())

    # Method with normalizing image
    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], 4), batch_size=self.minibatch_size)
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

    def get_action(self, state, available_actions, expl_rate):
        #  This means that every value within the range [0, 1) has an equal probability of being chosen.
        # NOTE: I don't know if the original person who created this intended to leave out a maxval of 1.
        # The numbers are chosen between [0,1)
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
        
    def remember(self, state, action, reward, next_state, game_over):
        self.replay_memory.append((state, action, reward, next_state, game_over))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # NOTE: get_eps function taken from Atari game. Used to calculate epsilon value for epsilon-greedy policy based on an annealing schedule.
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
    
    # NOTE: update_main_model function taken from Atari game. Used to update main q-network.
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

        state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = [], [], [], [], []
        for index in indices_lst:
            (state, action, reward, next_state, game_over) = self.replay_memory[index]
            state_batch.append(tf.constant(state, tf.float32))
            action_batch.append(tf.constant(action, tf.string))
            reward_batch.append(tf.constant(reward, tf.float32))
            next_state_batch.append(tf.constant(next_state, tf.float32))
            game_over_batch.append(tf.constant(game_over, tf.bool))
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
        return concatenated_state_tensor, tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), concatenated_next_state_tensor, tf.stack(game_over_batch, axis=0)

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

    def train_agent(self, maze: Maze, num_episodes = 1e7):
        for episode in range(num_episodes):
            self.cur_stacked_images.clear()
            time_step = 0
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1). NOTE: We do not downsize our image in preprocessing just yet.
            init_state = maze.reset(time_step)
            state = self.preprocess_image(time_step, init_state)
            # episode_step = 0
            episode_score = 0.0
            game_over = False
            while not game_over:
                # From Google article pseudocode line 5: With probability epsilon select a random action a_t
                expl_rate = self.get_eps(current_step=time_step)
                available_actions = maze.get_available_actions()
                action = self.get_action(state, available_actions, expl_rate)
                # NOTE: Should I only add to time_step if step was valid?
                time_step += 1
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_over) = maze.take_action(action, time_step)
                episode_score += reward
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = self.preprocess_image(time_step, next_state_img)
                # From Google article pseudocode line 8: Store transition/experience in D(replay memory)
                self.remember(state, action, reward, next_state, game_over)
                state = next_state
                # From Google article pseudocode line 9: Sample random minibatch of transitions/experiences from D
                if (time_step % self.agent_history_length == 0) and (time_step > self.replay_start_size):
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.generate_minibatch_samples()
                    self.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                # From Google article pseudocode line 10: if episode terminates at step j+1
                if game_over:
                    print('Game Over.')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(time_step))
                    # break
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

                # From Google article pseudocode line 12: Every C steps reset Q^hat = Q
                if ((episode % self.update_target_network_freq == 0) and (episode > self.replay_start_size)):
                    self.update_target_model()

