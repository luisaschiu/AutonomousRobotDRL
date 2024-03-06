import numpy as np
import tensorflow as tf
from collections import deque
import random
from class_maze import Maze
from class_DQN import DQN
from tensorflow.keras import initializers, models, optimizers, metrics, losses
from tensorflow.keras.layers import  Conv2D, Flatten, Dense, Lambda, Input
from tensorflow.keras.models import Model
import cv2 as cv
import itertools
from random import shuffle, randrange


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
    init_state = maze.reset(0)
    network = DQN((init_state.shape))
    history = network.train_agent(maze,100)