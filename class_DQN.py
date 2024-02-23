import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.python.keras import layers, initializers, models, optimizers, metrics, losses
import cv2 as cv

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
        self.model = self._build_model()
        
        self.target_model = self._build_model()
        self.update_target_network_freq = 1000
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        # NOTE: Bottom line of code might be redundant, unless I include the feature where I load existing trained weights from a given file
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        model = models.Sequential()
        init = initializers.VarianceScaling(scale=2.0)
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        model.add(layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=init, input_shape=(None, self.state_size[0], self.state_size[1], self.agent_history_length)))
        model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=init))
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=init))
        model.add(layers.Flatten())
        # Fully connected layer with 512 units, ReLU activation
        model.add(layers.Dense(512, activation='relu', kernel_initializer=init))
        # Output layer
        model.add(layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        return model

    def get_action(self, state, available_actions, expl_rate):
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            return random.choice(available_actions)
            # return random.randrange(self.action_size)
        else:
            max_val_idx =  np.argmax(self.model.predict(state)[0])
            # TODO: Fix this portion of code, because max_val_idx may be a number larger than the available_actions given.
            # Maybe try doing self.model.predict, then seeing the list of values that come out and sort it that way
            return available_actions[max_val_idx]
        
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

        if current_step < self.replay_start_size:
            eps = self.init_exploration_rate
        elif self.replay_start_size <= current_step and current_step < self.final_exploration_frame:
            eps = (self.final_exploration_rate - self.init_exploration_rate) / (self.final_exploration_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_exploration_rate
        elif self.final_exploration_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_exploration_rate) / (terminal_eps_frame - self.final_exploration_frame) * (current_step - self.final_exploration_frame) + self.final_exploration_rate
        else:
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
            next_state_q = self.target_model(next_state_batch)
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
            while True:
                # If replay memory is full and has hit it's maximum capacity, find a random index in the range: history length and memory_capacity
                if self.agent_history_length == self.replay_memory_capacity:
                    index = np.random.randint(low=self.agent_history_length, high=self.replay_memory_capacity, dtype=np.int32)
                else:
                # If replay memory isn't full yet, sample from existing replay memory
                    index = np.random.randint(low=self.agent_history_length, high=cur_memory_size, dtype=np.int32)
                # If any cases are terminal, disregard and keep looking for a new random index to add onto the list
                    if np.any([(sample[4] == True) for sample in self.replay_memory[index - self.agent_history_length:index]]):
                        continue
                indices_lst.append(index)
                break
        state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = [], [], [], [], []
        for index in indices_lst:
            (state, action, reward, next_state, game_over) = self.replay_memory[index]
            state_batch.append(tf.constant(state, tf.float32))
            action_batch.append(tf.constant(action, tf.int32))
            reward_batch.append(tf.constant(reward, tf.float32))
            next_state_batch.append(tf.constant(next_state, tf.float32))
            game_over_batch.append(tf.constant(game_over, tf.bool))
        return tf.stack(state_batch, axis=0), tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), tf.stack(next_state_batch, axis=0), tf.stack(game_over_batch, axis=0)

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
        game_over = False
        for episode in range(num_episodes):
            self.cur_stacked_images.clear()
            time_step = 0
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1). NOTE: We don't have image preprocessing implemented just yet.
            init_state = maze.reset(time_step)
            state = self.preprocess_image(time_step, init_state)
            # episode_step = 0
            episode_score = 0.0
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
                next_state = self.preprocess_image(time_step, next_state_img)
                # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                # next_state = (state, action, next_state_img)
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
                    break
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

