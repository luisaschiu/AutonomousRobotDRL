import numpy as np
from class_maze import Maze
import AruCo_functions

# Axis system will not follow numpy array's. Image pixels are read x horizontal axis (positive to the right),
# y vertical axis (positive down). Internally, code will be edited such that it follows the x horizontal, 
# y vertical axis system to prevent confusion. I chose to prioritize the image pixels over the numpy array, 
# because images will be input into the neural network, and the numpy array is just used to simulate and visually
# showcase what's going on.
maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]])
marker_filepath = "images/marker8.jpg"
maze = Maze(maze_array, marker_filepath, (0,0), (2,3), 2)

# Test AruCo functions, edited from computer vision project to follow the axes defined (x horizontal, y vertical)
# Maze.show(maze)
Maze.generate_img(maze)
# AruCo_functions.arucode_location()
print(maze.traversed)
Maze.move_robot(maze, "DOWN")
print(maze.traversed)
# Maze.show(maze)
Maze.generate_img(maze)


# Testing overall functions for moving robot and maze traversing
# Maze.show(maze)
# print(maze.traversed)
# Maze.move_robot(maze, "DOWN")
# print(maze.traversed)
# Maze.show(maze)
# Maze.move_robot(maze, "DOWN") # Yield wall detected error
# print(maze.traversed)
# Maze.move_robot(maze, "RIGHT")
# print(maze.traversed)
# Maze.move_robot(maze, "RIGHT")
# Maze.move_robot(maze, "RIGHT")
# print(maze.traversed)
# Maze.move_robot(maze, "RIGHT") # Yield maze edge detected error
# print(maze.traversed)
# Maze.show(maze)
# # Test resetting the maze
# print("Resetting Maze")
# Maze.reset(maze)
# print(maze.traversed)
# Maze.show(maze)