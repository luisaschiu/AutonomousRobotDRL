import numpy as np
from class_maze import Maze
import ArUco_functions
import cv2 as cv
import os
import glob

for filename in glob.glob('robot_steps/*.jpg'):
    os.remove(filename)
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
maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
# maze.show()

init= maze.reset(0)

print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action("DOWN", 1)
print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action( "RIGHT", 2)
print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action("RIGHT", 3)
print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action("DOWN", 4)
print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action("DOWN", 5)
print(maze.get_available_actions())
# print(maze.traversed)
maze.take_action("RIGHT", 6)
print(maze.get_available_actions())