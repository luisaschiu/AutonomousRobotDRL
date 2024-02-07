import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.python.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size=4, agent_history_length = 4, replay_memory_size=10000000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_history_length = agent_history_length
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.gamma = gamma # Also known as discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation = 'relu', padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation = 'relu', padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='same'))
        model.add(layers.Flatten())
        # Fully connected layer with 512 units, ReLU activation
        model.add(layers.Dense(512, activation='relu'))
        # TODO: Figure out output layer of 4 actions
        model.add(layers.Dense(self.action_size, activation='linear'))
        return model
        # model.compile(optimizer=optimizers.SGD(learning_rate= self.learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
    
    def train_agent(robot, maze, num_episodes = 100):
        pass