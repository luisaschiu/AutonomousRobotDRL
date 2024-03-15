import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.keras import initializers, models, optimizers, metrics, losses
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input, Rescaling
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
        self.replay_start_size = 10
        # self.replay_start_size = 1e4
        self.discount_factor = 0.99 # Also known as gamma
        self.minibatch_size = 32
        self.max_steps_per_episode = 20 # TODO: Chosen arbitrarily right now, make sure you change this as needed
        self.update_target_network_freq = 1000
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.model = self.build_model()
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        self.init_exploration_rate = 1.0 # Exploration rate, also known as epsilon
        self.final_exploration_rate = 0.1
        # self.final_exploration_frame = 1e6
        self.final_exploration_frame = 200
        self.learning_rate = 0.001
        self.target_model = models.clone_model(self.model)
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        self.target_model.set_weights(self.model.get_weights())
        # optimizer = optimizers.RMSProp(learning_rate= self.learning_rate),loss='mse') # From paper info, maybe misinterpreted?
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-6)
        self.loss_metric = metrics.Mean(name="loss")
        self.Q_value_metric = metrics.Mean(name="Q_value")

    # Method with normalizing image
    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        # init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], self.agent_history_length), batch_size=self.minibatch_size)
        # input_layer = Input(shape = (389, 398, 4))
        normalized_input = Rescaling(scale=1.0/255.0)(input_layer)
        # normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
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

    def remember(self, state, action, reward, next_state, game_over, next_state_available_action):
        self.replay_memory.append((state, action, reward, next_state, game_over, next_state_available_action))

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
            print("In if statement")
            eps = self.init_exploration_rate
        # If the robot has taken enough steps before replaying old memories and updating the main model (greater than or equal to 
        # self.replay_start_size) and it is not at the last frame in which we want it to explore less.
        elif self.replay_start_size <= current_step and current_step < self.final_exploration_frame:
            print("In 1st elif statement")
            eps = (self.final_exploration_rate - self.init_exploration_rate) / (self.final_exploration_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_exploration_rate
        # If the robot has taken enough steps as it gets closer to the final frames before it needs to be terminated to prevent over exploring
        elif self.final_exploration_frame <= current_step and current_step < terminal_eps_frame:
            print("In 2nd elif statement")
            eps = (terminal_eps - self.final_exploration_rate) / (terminal_eps_frame - self.final_exploration_frame) * (current_step - self.final_exploration_frame) + self.final_exploration_rate
        else:
            # Right now, self.final_exploration_rate = 0.01. terminal_eps is 0.01. This means epsilon is very low, and 
            # there is a very low chance of exploration.
            print("In else statement")
            eps = terminal_eps
        return eps
    
    def get_action(self, state, available_actions, expl_rate):
        # #  This means that every value within the range [0, 1) has an equal probability of being chosen.
        actions_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            # Filter available actions
            valid_actions = [action for action, is_available in zip(actions_list, available_actions) if is_available]
            return random.choice(valid_actions)
        else:
            array=self.model.predict(state)
            # Copy array so we don't alter the original q-value array in case we want to look at it
            print(array)
            masked_qval_array = np.where(np.array(available_actions) == 1, array, float('-inf'))
            print(masked_qval_array)
            max_val_index = np.argmax(np.max(masked_qval_array, axis=0))
            print(max_val_index)
            return actions_list[max_val_index]    


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
        mean = tf.math.reduce_mean(loss_val)
        # print(mean.numpy())
        return mean.numpy()

if __name__ == "__main__":
    # pass
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
    total_step = 0
    episode_step = 0
    episode_score = 0
    init_state = maze.reset(episode_step)
    state = network.preprocess_image(episode_step, init_state)
   
    # Test lambda layer normalizing the input layer in build_model()
    print("Before normalizing, input data:")
    print(state)
    input_layer = Input(shape = (389, 389, 4), batch_size=32)
    normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
    temp_model = models.Model(inputs=input_layer, outputs=normalized_input)
    lambda_output = temp_model.predict(state)
    print("After normalizing, output data:")
    print(lambda_output)


    # # Initial parameters: create maze
    # # Testing one run of the train_agent code:
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
    # total_step = 0
    # episode_step = 0
    # episode_score = 0
    # init_state = maze.reset(episode_step)
    # state = network.preprocess_image(episode_step, init_state)
    # game_over = False
    # while not game_over:
    #     # From Google article pseudocode line 5: With probability epsilon select a random action a_t
    #     expl_rate = network.get_eps(total_step)
    #     state_available_actions = maze.get_available_actions()
    #     # print(state_available_actions)
    #     action = network.get_action(state, state_available_actions, expl_rate)
    #     total_step += 1
    #     episode_step += 1
    #     # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
    #     (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    #     episode_score += reward
    #     next_state_available_actions = maze.get_available_actions()
    #     # next_state_available_actions_filtered = [0 if x is None else x for x in next_state_available_actions]
    #     # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
    #     next_state = network.preprocess_image(episode_step, next_state_img)
    #     # From Google article pseudocode line 8: Store transition/experience in D(replay memory)
    #     network.remember(state, action, reward, next_state, game_over, next_state_available_actions)
    #     state = next_state
    #     if (total_step % network.agent_history_length == 0) and (total_step > network.replay_start_size):
    #         print("Generating minibatch and updating main model")
    #         state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch = network.generate_minibatch_samples()
    #         # print("next_state_available_actions_batch")
    #         # print(next_state_available_actions_batch)
    #         network.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch)
    #     if episode_step == network.max_steps_per_episode:
    #         game_over = True
    #     # From Google article pseudocode line 10: if episode terminates at step j+1
    #     if game_over:
    #         print('Game Over.')
    #         print('Episode Num: ' + str(0) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
    #         # break
    #     # If episode does not terminate... continue onto last lines of pseudocode
    #     # From Google article pseudocode line 11: Perform a gradient descent step (done in update_main_model)
            
    #     # From Google article pseudocode line 12: Every C steps reset Q^hat = Q
    #     if ((total_step % network.update_target_network_freq == 0) and (total_step > network.replay_start_size)):
    #         network.update_target_model()
    #     print("ep step: ", episode_step)


    # Test finding mean of tensor:
    # test_tensor = tf.constant([0.1, 0.4, 0.6, 0.8, 0.3689], tf.float32)
    # mean = tf.math.reduce_mean(test_tensor)
    # print(mean.numpy())


    # # Test get_action() logic:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # marker_filepath = "images/marker8.jpg"
    # maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
    # network = DQN((389, 389))
    # model = network.build_model()
    # init_state = maze.reset(0)
    # state = network.preprocess_image(0, init_state)
    # available_actions = maze.get_available_actions()
    # expl_rate = 1.0
    # action = network.get_action(state, available_actions, expl_rate)
    # print(action)
    # (next_state_img, reward, game_over) = maze.take_action(action, 1)
    # expl_rate = 0.3
    # next_state = network.preprocess_image(1, next_state_img)
    # available_actions = maze.get_available_actions()
    # action = network.get_action(state, available_actions, expl_rate)
    # print(action)
    # state = next_state


    # # Test finding highest max_val in array and choosing next best action(max q value) if the "best_action" is unavailable for get_action().
    # actions_list = ["UP", "DOWN", "LEFT", "RIGHT"]
    # # By random choice:
    # print("random choice made: ")
    # # Filter available actions
    # valid_actions = [action for action, is_available in zip(actions_list, available_actions) if is_available]
    # print("valid_action: ", valid_actions)
    # print(random.choice(valid_actions))
    # # By model.predict (generating random q value array)
    # test_array = np.array([0, 1, 0, 1])
    # rand_qval_array = np.random.uniform(low=-1, high=1, size=(1, 4))
    # print(rand_qval_array)
    # masked_qval_array = np.where(test_array == 1, rand_qval_array, float('-inf'))
    # print(masked_qval_array)
    # max_col_index = np.argmax(np.max(masked_qval_array, axis=0))
    # print(max_col_index)


    # # Test adjusting tensor to look for best q-value in tensor with an available action
    # test_tensor = tf.constant([[0, 0, 0, 1,],
    #                 [1, 1, 0, 0],
    #                 [0, 1, 1, 1],
    #                 [1, 1, 0, 0],
    #                 [0, 1, 0, 1],
    #                 [1, 1, 0, 0],
    #                 [0, 1, 1, 1]], tf.int32)
    # print(test_tensor)
    # rand_qval_tensor = tf.random.uniform(shape=(7, 4), minval=-1, maxval=1)
    # print(rand_qval_tensor)
    # masked_qval_tensor = tf.where(test_tensor == 1, rand_qval_tensor, tf.constant(float('-inf'), shape=rand_qval_tensor.shape))
    # print(masked_qval_tensor)
    # largest_values = tf.math.reduce_max(masked_qval_tensor, axis=1)
    # print(largest_values)


    # # Test loss value tensor shape in update_main_model()
    # # main_q =[[0, 1], [0, 0], [0, 3]]
    # # expected_q = [[0.6, 0.4], [0.4, 0.6], [7, 3]]
    # main_q = [2, 4, 6, 8, 8]
    # expected_q = [5, 6, 4, 6, 4]
    # main_q_tensor = tf.expand_dims(tf.convert_to_tensor(main_q), axis = 1)
    # exp_q_tensor=tf.expand_dims(tf.convert_to_tensor(expected_q), axis = 1)
    # loss = losses.Huber(reduction=losses.Reduction.NONE)
    # loss_val = loss(tf.stop_gradient(exp_q_tensor), main_q_tensor)
    # print(loss_val)
    # print(loss_val.shape)


    # # Test one hot encoding for update_main_model()
    # action_size = 4
    # action_batch = ["UP", "DOWN", "UP", "LEFT", "LEFT", "RIGHT", "DOWN", "DOWN", "RIGHT"]
    # action_tensor = tf.constant(action_batch, tf.string)
    # print(action_tensor)
    # # unique_actions = ["UP", "DOWN", "LEFT", "RIGHT"]  # Get unique actions
    # # print(unique_actions)
    # unique_actions = tf.constant(["UP", "DOWN", "LEFT", "RIGHT"])  # Get unique actions as a TensorFlow constant
    # action_indices = tf.argmax(tf.cast(tf.equal(unique_actions[:, tf.newaxis], action_tensor), tf.int32), axis=0)
    # print(action_indices)
    # action_one_hot = tf.one_hot(action_indices, depth=action_size, on_value=1.0, off_value=0.0)
    # print(action_one_hot)
    # # # Approach using numpy (Not able to use for @tf.function)
    # # action_batch_np = np.array(action_batch)
    # # action_indices = [action_to_index[action] for action in action_batch_np]
    # # print(action_indices)
    # # # Step 3: Perform one-hot encoding
    # # action_one_hot = tf.one_hot(action_indices, depth=action_size, on_value=1.0, off_value=0.0)
    # # print(action_one_hot)


    # # Test lambda layer normalizing the input layer in build_model()
    # print("Before normalizing, input data:")
    # print(state)
    # input_layer = Input(shape = (389, 389, 4), batch_size=32)
    # normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
    # temp_model = models.Model(inputs=input_layer, outputs=normalized_input)
    # lambda_output = temp_model.predict(state)
    # print("After normalizing, output data:")
    # print(lambda_output)


    # # Test .predict for network.model and model.target_model
    # (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    # episode_score += reward
    # next_state = network.preprocess_image(episode_step, next_state_img)
    # network.remember(state, action, -1, next_state, game_over)
    # action = "RIGHT"
    # (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    # episode_score += reward
    # next_state = network.preprocess_image(episode_step, next_state_img)
    # network.remember(state, action, -1, next_state, game_over)
    # action = "RIGHT"
    # (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    # episode_score += reward
    # next_state = network.preprocess_image(episode_step, next_state_img)
    # network.remember(state, action, -1, next_state, game_over)
    # action = "LEFT"
    # (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    # episode_score += reward
    # next_state = network.preprocess_image(episode_step, next_state_img)
    # action = "RIGHT"
    # (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
    # episode_score += reward
    # next_state = network.preprocess_image(episode_step, next_state_img)
    # network.remember(state, action, -1, next_state, game_over)
    # state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = network.generate_minibatch_samples()
    # test = network.target_model.predict(next_state_batch)
    # print(test)


    # # Test updating target_model weights, printing the weights of single layer or all layers
    # network.model.summary()
    # layer_index = 7
    # layer_weights = network.model.layers[layer_index].get_weights()
    # print(f"Weights of layer {layer_index}:")
    # print(layer_weights)
    # # Before updating target model:
    # layer_weights = network.target_model.layers[layer_index].get_weights()
    # print(f"Weights of layer {layer_index}:")
    # print(layer_weights)
    # network.target_model.summary()
    # # After updating target model:
    # network.update_target_model()
    # layer_weights = network.target_model.layers[layer_index].get_weights()
    # print(f"Weights of layer {layer_index}:")
    # print(layer_weights)
    # network.target_model.summary()

    # # Print weights of each layer
    # weights = network.model.get_weights()
    # for i, layer_weights in enumerate(weights):
    #     print(f"Layer {i} weights:")
    #     print(layer_weights)
    # # Before updating target model:
    # target_weights = network.target_model.get_weights()
    # for i, layer_weights in enumerate(target_weights):
    #     print(f"Layer {i} weights:")
    #     print(layer_weights)
    # # After updating target model:
    # network.update_target_model()
    # target_weights = network.target_model.get_weights()
    # for i, layer_weights in enumerate(target_weights):
    #     print(f"Layer {i} weights:")
    #     print(layer_weights)

    # # Test get_eps()
    # print(network.get_eps(1))
    # print(network.get_eps(100))
    # print(network.get_eps(200))
    # print(network.get_eps(200000))


    # # Test logic for finding valid_indices of minibatches in generate_minibatch_samples()
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
    # # Start of logic for general deque slicing
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


    # # Test inputting varying batch size inputs in .predict for model
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


    # # Test output shape of batches that are not state or next_state in generate_minibatch_samples()
    # game_over_batch = []
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_batch.append(tf.constant(False, tf.bool))
    # game_over_batch.append(tf.constant(True, tf.bool))
    # game_over_tensor = tf.stack(game_over_batch, axis=0)
    # print(game_over_tensor)

