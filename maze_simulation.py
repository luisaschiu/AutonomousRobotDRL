import numpy as np
from class_maze import Maze


maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]])

maze = Maze(maze_array, (0,0), (3,3))
Maze.show(maze)
Maze.move_robot(maze, "DOWN")
Maze.show(maze)
