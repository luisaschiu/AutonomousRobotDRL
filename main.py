import numpy as np
from class_maze import Maze
from class_DQN import DQN

# Initial parameters: create maze
# Testing train_agent:
maze_array = np.array(
[[0.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[1.0, 1.0, 0.0, 1.0],
[0.0, 1.0, 0.0, 0.0]])
marker_filepath = "images/marker8.jpg"
maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180, False)
network = DQN((389, 389), False)
network.train_agent(maze, 25)

# maze/dqn now have a realTimeFlag attribute
# default is True, can be set in both classes