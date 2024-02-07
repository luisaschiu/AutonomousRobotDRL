import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from tensorflow.python.keras import layers, models, optimizers

class DQN:
    def __init__(self, state_size, action_size=4, replay_memory_size=10000000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=32):
        # NOTE: Is state_size a (m,n) for a mxn maze or the image size?
        self.state_size = state_size
        # NOTE: Action_size will change, when do I declare available actions? After running through NN, or after?
        self.action_size = action_size
        # From Google article pseudocode line 1: Initialize replay memory D to capacity N
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.gamma = gamma # Also known as discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_timesteps = 20 # TODO: Chosen arbitrarily right now, make sure you change this as needed
        self.model = self._build_model()
        self.target_model = self._build_model()
        # From Google article pseudocode line 3: Initialize action-value function Q^hat(target network) with same weights as Q
        # NOTE: Might be redundant, unless I include the feature where I load existing trained weights
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # NOTE: Random weights are initialized, might want to include an option to load weights from a file to continue training
        # From Google article pseudocode line 2: Initialize action-value function Q with random weights
        model = models.Sequential()
        init = layers.initializers.RandomNormal(mean=0.0, stddev=0.1)  # Adjust mean and stddev as needed
        model.add(layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same', kernel_initializer=init, input_shape=self.state_size))
        model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same', kernel_initializer=init))
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same', kernel_initializer=init))
        model.add(layers.Flatten())
        # Fully connected layer with 512 units, ReLU activation
        model.add(layers.Dense(512, activation='relu', kernel_initializer=init))
        # Output layer
        model.add(layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        model.compile(optimizer=optimizers.RMSProp(learning_rate= self.learning_rate),loss='mse', metrics=['accuracy'])
        # NOTE: As stated by keras documentation:
        # Metric values are displayed during fit() and logged to the History object returned by fit(). They are also returned by model.evaluate().
        return model

    # NOTE: Else statement gives the index of the maximum value in the output of the neural network model.
    def choose_action(self, state):
        actions = ["UP", "DOWN", "RIGHT", "LEFT"]
        # Eventually, won't be constant 4 actions. Will filter out.
        if np.random.rand() <= self.epsilon:
            return random.choice(actions)
            # return random.randrange(self.action_size)
        else:
            max_val_idx =  np.argmax(self.model.predict(state)[0])
            return actions[max_val_idx]
        
    def remember(self, state, action, reward, next_state, game_over):
        self.replay_memory.append((state, action, reward, next_state, game_over))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def execute_and_observe(self):
        pass

    def train_agent(self, maze: Maze, num_episodes = 100):
        for _ in range(num_episodes):
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1). NOTE: We don't have preprocessing implemented just yet.
            cur_state = maze.generate_img()
            # TODO: Look out for where I want to keep track of timesteps, either here or in class_maze.
            for t in range(self.max_timesteps):
                # From Google article pseudocode line 5: With probability epsilon select a random action a_t
                action = self.choose_action(cur_state)
                # From Google article pseudocode line 6: Execute action a_t in emulator and observe reward rt and image x_t+1
                (next_state_img, reward, game_status) = maze.take_action(action)
                # Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = (cur_state, action, next_state_img)
                # Store transition/experience in D(replay memory)
                # NOTE: game_over is not a parameter they save in the replay_memory in the article
                self.remember(cur_state, action, reward, next_state, game_status)
