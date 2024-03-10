import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.keras import initializers, models, optimizers, metrics, losses
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
import cv2 as cv
import itertools
import time

# Pre-compute unique_actions
unique_actions = tf.constant(["UP", "DOWN", "LEFT", "RIGHT"])

class DQN:
    def __init__(self, state_size, size):
        self.state_size = state_size
        self.action_size = 4
        self.replay_memory_capacity=10000000
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.replay_start_size = size * 2
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.final_exploration_frame = size**2
        self.learning_rate = 1e-4
        self.minibatch_size = 10
        self.max_steps_per_episode = size**2
        self.win_history = []
        self.agent_history_length = 4
        self.model = self.build_model()
        self.target_model = models.clone_model(self.model)
        self.update_target_network_period = 10
        self.cur_stacked_images = deque(maxlen=self.agent_history_length)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-8)
        self.loss_metric = metrics.Mean(name="loss")
        self.Q_value_metric = metrics.Mean(name="Q_value")

    def build_model(self):
        input_layer = Input(shape = (self.state_size[0], self.state_size[1], self.agent_history_length), batch_size=self.minibatch_size)
        normalized_input = Lambda(lambda x: x / 255.0)(input_layer)
        conv1 = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(normalized_input)
        conv2 = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=initializers.VarianceScaling(scale=2.0))(conv2)
        flatten = Flatten()(conv3)
        dense1 = Dense(512, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.0))(flatten)
        output_layer = Dense(4, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.0))(dense1)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def get_action(self, state, available_actions, expl_rate):
        actions_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < expl_rate:
            valid_actions = [action for action, is_available in zip(actions_list, available_actions) if is_available]
            return random.choice(valid_actions)
        else:
            array=self.model.predict(state)
            masked_qval_array = np.where(np.array(available_actions) == 1, array, float('-inf'))
            max_val_index = np.argmax(np.max(masked_qval_array, axis=0))
            return actions_list[max_val_index]
        
    def remember(self, state, action, reward, next_state, game_over, next_state_available_action):
        self.replay_memory.append((state, action, reward, next_state, game_over, next_state_available_action))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
            eps = self.epsilon_start
        elif self.replay_start_size <= current_step and current_step < self.final_exploration_frame:
            eps = (self.epsilon_end - self.epsilon_start) / (self.final_exploration_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.epsilon_start
        elif self.final_exploration_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.epsilon_end) / (terminal_eps_frame - self.final_exploration_frame) * (current_step - self.final_exploration_frame) + self.epsilon_end
        else:
            eps = terminal_eps
        return eps
    

    @tf.function
    def update_main_model(self, state_batch, action_batch, reward_batch, next_state_batch, game_over_batch, next_state_available_actions_batch):
        with tf.GradientTape() as tape:
            next_state_q = self.target_model(next_state_batch)
            masked_q_tensor = tf.where(next_state_available_actions_batch == 1, next_state_q, tf.constant(float('-inf'), shape=next_state_q.shape))
            next_state_max_q = tf.math.reduce_max(masked_q_tensor, axis=1)
            expected_q = reward_batch + self.gamma * next_state_max_q * (1.0 - tf.cast(game_over_batch, tf.float32))
            action_indices = tf.argmax(tf.cast(tf.equal(unique_actions[:, tf.newaxis], action_batch), tf.int32), axis=0)
            action_one_hot = tf.one_hot(action_indices, depth=self.action_size, on_value=1.0, off_value=0.0)
            main_q = tf.reduce_sum(self.model(state_batch) * action_one_hot, axis=1)
            main_q_dim = tf.expand_dims(main_q, axis = 1)
            expected_q_dim = tf.expand_dims(expected_q, axis = 1)
            loss = losses.Huber(reduction=losses.Reduction.NONE)
            loss_val = loss(tf.stop_gradient(expected_q_dim), main_q_dim)

        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        for var, grad in zip(self.model.trainable_variables, clipped_gradients):
            var.assign_sub(self.learning_rate * grad)

        self.loss_metric.update_state(loss_val)
        self.Q_value_metric.update_state(main_q)

        return loss_val

    def generate_minibatch_samples(self):
        start_time = time.time()
        indices_lst = np.zeros(self.minibatch_size, dtype=np.int32)
        cur_memory_size = len(self.replay_memory)
        count = 0
        while count < self.minibatch_size:
            if cur_memory_size == self.replay_memory_capacity:
                index = np.random.randint(low=self.agent_history_length, high=(self.replay_memory_capacity+1), dtype=np.int32)
            else:
                index = np.random.randint(low=self.agent_history_length, high=(cur_memory_size+1), dtype=np.int32)
            if not any(item[4] for item in list(self.replay_memory)[index-self.agent_history_length:index]):
                indices_lst[count] = index - 1
                count += 1

        state_batch = [tf.constant(self.replay_memory[index][0], tf.float32) for index in indices_lst]
        action_batch = [tf.constant(self.replay_memory[index][1], tf.string) for index in indices_lst]
        reward_batch = [tf.constant(self.replay_memory[index][2], tf.float32) for index in indices_lst]
        next_state_batch = [tf.constant(self.replay_memory[index][3], tf.float32) for index in indices_lst]
        game_over_batch = [tf.constant(self.replay_memory[index][4], tf.bool) for index in indices_lst]
        next_state_available_actions_batch = [tf.constant(self.replay_memory[index][5], tf.int32) for index in indices_lst]

        concatenated_state_tensor = tf.concat(state_batch, axis=0)
        concatenated_next_state_tensor = tf.concat(next_state_batch, axis=0)
        end_time = time.time()
        print(f"Execution time of generate_minibatch_samples() is {end_time - start_time} seconds")
        return concatenated_state_tensor, tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), concatenated_next_state_tensor, tf.stack(game_over_batch, axis=0), tf.stack(next_state_available_actions_batch, axis=0)


    def preprocess_image(self, time_step, new_image):
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        
        # Ensure the image is square and of size state_size x state_size
        new_image = cv.resize(new_image, (self.state_size[0], self.state_size[1]))
        
        if time_step == 0:
            self.cur_stacked_images = [new_image] * 4
        else:
            self.cur_stacked_images.append(new_image)
            self.cur_stacked_images.pop(0)
        
        # Ensure all images in cur_stacked_images are square and of size state_size x state_size
        for i, img in enumerate(self.cur_stacked_images):
            self.cur_stacked_images[i] = cv.resize(img, (self.state_size[0], self.state_size[1]))
        
        tensor = tf.constant(self.cur_stacked_images, tf.float32)
        tensor_transposed = tf.transpose(tensor, [1, 2, 0])
        tensor_batch = tf.expand_dims(tensor_transposed, axis=0)
        
        return tensor_batch
    

    def train_agent(self, maze: Maze, num_episodes = 1e7):
        total_step = 0
        scores=[]
        for episode in range(num_episodes):
            self.cur_stacked_images.clear()
            episode_step = 0
            episode_score = 0.0
            game_over = False
            init_state = maze.reset(episode_step)
            state = self.preprocess_image(episode_step, init_state)

            while not game_over:
                expl_rate = self.get_eps(total_step)
                available_actions = maze.get_available_actions()
                action = self.get_action(state, available_actions, expl_rate)
                total_step += 1
                episode_step += 1
                (next_state_img, reward, game_over) = maze.take_action(action, episode_step)
                episode_score += reward
                next_state_available_actions = maze.get_available_actions()
                next_state = self.preprocess_image(episode_step, next_state_img)
                self.remember(state, action, reward, next_state, game_over, next_state_available_actions)
                state = next_state
                if (total_step % self.agent_history_length == 0) and (total_step > self.replay_start_size):
                    print("Generating minibatch and updating main model")
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch = self.generate_minibatch_samples()
                    start_time = time.time()
                    loss = self.update_main_model(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, next_state_available_actions_batch)
                    end_time = time.time()
                    print(f"Execution time of the function updata_main_model() is {end_time - start_time} seconds")
                    print(f"Loss = {np.mean(loss)}\n")
                if episode_step == self.max_steps_per_episode:
                    game_over = True
                    maze.num_traversed =0
                if game_over:
                    print('Game Over.')
                    print('Episode Num: ' + str(episode) + ', Episode Rewards: ' + str(episode_score) + ', Num Steps Taken: ' + str(episode_step))
                    scores.append((episode,episode_score))
                if ((total_step % self.update_target_network_period == 0) and (total_step > self.replay_start_size)):
                    self.update_target_model()
            
        return scores
                