import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.keras import initializers, models, optimizers, metrics, losses
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
from tensorflow.keras.models import Model
import cv2 as cv
import itertools
from random import shuffle, randrange

class DQN:
    def __init__(self, state_size):
        # State size is the image size
        self.state_size = state_size
        self.action_size = 4
        self.replay_memory_capacity=10000000
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.replay_start_size = 10
        self.discount_factor = 0.99 # Also known as gamma
        self.minibatch_size = 32
        self.max_steps_per_episode = 1000 # TODO: Chosen arbitrarily right now, make sure you change this as needed
        self.update_target_network_freq = 1000 # Update Rate Tau 0.005 = 200
        self.agent_history_length = 4 # Number of images from each timestep stacked
        self.model = self.build_model()
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        self.init_exploration_rate = 0.9 # Exploration rate, also known as epsilon start
        self.final_exploration_rate = 0.05 # Known as Epsilon End
        self.final_exploration_frame = 200
        self.learning_rate = 1e-4
        self.target_model = models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, epsilon=1000)
        self.loss_metric = metrics.Mean(name="loss")
        self.Q_value_metric = metrics.Mean(name="Q_value")
    

    # Method with normalizing image
    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], self.agent_history_length), batch_size=self.minibatch_size)
        normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
        conv1 = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(normalized_input)
        conv2 = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv2)
        flatten = Flatten()(conv3)
        dense1 = Dense(512, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.0))(flatten)
        output_layer = Dense(4, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.0))(dense1)
        
        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Print the model summary
        model.summary()
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
            terminal_flag = any(item[4] for item in sliced_deque)
            if not terminal_flag:
                # Since slicing the deque doesn't consider the last index, I have to offset the index by 1.
                # Slice notation [start:stop] extracts elements from the index start up to, but not including, the index stop.
                indices_lst.append(index-1)

        state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = zip(*[self.replay_memory[index] for index in indices_lst])
        state_batch = tf.reshape(tf.stack([tf.constant(state, tf.float32) for state in state_batch], axis=0), (-1, 389, 389, 4))
        action_batch = tf.stack([tf.constant(action, tf.string) for action in action_batch], axis=0)
        reward_batch = tf.stack([tf.constant(reward, tf.float32) for reward in reward_batch], axis=0)
        next_state_batch = tf.reshape(tf.stack([tf.constant(next_state, tf.float32) for next_state in next_state_batch], axis=0), (-1, 389, 389, 4))
        game_over_batch = tf.stack([tf.constant(game_over, tf.bool) for game_over in game_over_batch], axis=0)

        return state_batch, action_batch, reward_batch, next_state_batch, game_over_batch

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
        #  This means that every value within the range [0, 1) has an equal probability of being chosen.
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            not_none_idx = [index for index, action in enumerate(available_actions) if action is not None]
            random_idx = random.choice(not_none_idx)
            return available_actions[random_idx]
        else:
            array =self.model.predict(state)
            # Copy array so we don't alter the original q-value array in case we want to look at it
            array_copy = array.copy()
            best_action_idx = None
            while best_action_idx is None:
                max_idx = np.argmax(array_copy)
                col_idx = np.unravel_index(max_idx, array.shape)[1]
                if available_actions[col_idx] is not None:
                    best_action_idx = col_idx
                    break
                # If best_action is None, find the next largest q-value within the multidimensional array
                else:
                    array_copy.flat[max_idx] = np.iinfo(np.int32).min
            return available_actions[best_action_idx]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
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
            print("next_state_q")
            tf.print(next_state_q)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            print("next_state_max_q")
            tf.print(next_state_max_q)
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

        return loss_val

def make_maze(width, height, start, goal):
    # Create a maze filled with 1.0
    maze = np.ones((height, width))

    # The stack of visited cells
    stack = [start]

    # Mark the start cell as visited (carve a path) by setting it to 0.0
    maze[start[1], start[0]] = 0.0

    # Define the four possible directions to move
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        x, y = stack[-1]

        # Find all the unvisited neighbours
        neighbours = [(x + dx*2, y + dy*2) for dx, dy in directions
                      if (0 <= x + dx*2 < width) and (0 <= y + dy*2 < height) and maze[y + dy*2, x + dx*2]]

        if neighbours:
            # Choose a random neighbour
            nx, ny = neighbours[np.random.randint(len(neighbours))]

            # Carve a path to the neighbour by setting the cells to 0.0
            maze[ny, nx] = 0.0
            maze[y + (ny-y)//2, x + (nx-x)//2] = 0.0

            # Add the neighbour to the stack
            stack.append((nx, ny))
        else:
            # If there are no unvisited neighbours, backtrack
            stack.pop()

        # If the current cell is the goal, break the loop
        if (x, y) == goal:
            break

    return maze


if __name__ == "__main__":

    x_dim = 8
    y_dim = 8
    start = (0,0)
    goal = (x_dim-1,y_dim-1)
    maze_array = make_maze(x_dim, y_dim, start, goal)
    marker_filepath = "images/marker8.jpg"
    goal_filepath = "images/star.jpg"
    maze = Maze(maze_array, marker_filepath, goal_filepath, start, goal, 180)
    next_state_batch = []
    total_step = 0
    episode_step = 0
    episode_score = 0
    init_state = maze.reset(episode_step)
    network = DQN((init_state.shape[0], init_state.shape[1]))
    model = network.build_model()
    state = network.preprocess_image(episode_step, init_state)
    game_over = False
    game_expired = False

    while not game_over:
        # From Google article pseudocode line 5: With probability epsilon select a random action a_t
        expl_rate = network.get_eps(total_step)
        available_actions = maze.get_available_actions()
        action = network.get_action(state, available_actions, expl_rate)
        total_step += 1
        episode_step += 1
        # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
        (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
        episode_score += reward
        # From Google article pseudocode line 7: Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
        next_state = network.preprocess_image(episode_step, next_state_img)
        # From Google article pseudocode line 8: Store transition/experience in D(replay memory)
        network.remember(state, action, reward, next_state, game_over)
        state = next_state
        if (total_step % network.agent_history_length == 0) and (total_step > network.replay_start_size):
            print("Generating minibatch and updating main model")
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = network.generate_minibatch_samples()
            network.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
        if episode_step == network.max_steps_per_episode:
            game_expired = True
        # From Google article pseudocode line 10: if episode terminates at step j+1
        if game_over or game_expired:
            print(f"Game Over: You have {'won' if game_over else 'lost'}")
            print('Episode Num: ' + str(0) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
            break

        if ((total_step % network.update_target_network_freq == 0) and (total_step > network.replay_start_size)):
            network.update_target_model()
        print("ep step: ", episode_step)