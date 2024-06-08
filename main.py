import numpy as np
from class_maze import Maze
from class_DQN import DQN, show_pickle_figure, save_pickle_to_csv
import matplotlib.pyplot as plt
from itertools import count
import random
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
import os
import threading
import time
import pickle

def save_to_csv(data, file_path, headers=None):
    """
    Save data to a CSV file.

    Parameters:
        data: Data to be saved.
        file_path (str): Path to the CSV file.
        headers (list, optional): List of header names. Defaults to None.
    """
    mode = 'w' if headers else 'a'  # Use 'w' mode initially if headers are provided, otherwise use 'a' mode
    print("mode: ", mode)
    with open(file_path, mode, newline="") as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerow(data)

def TestCase0():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_size = 4
    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0, 0), (3,3), 270)
    maze1.generate_img(0)
    maze2 = Maze(maze_array1, marker_filepath, (0, 0), (3,3), 180)
    maze2.generate_img(1)
    network = DQN((120, 120), maze_size)
    # network.train_agent_static(maze1, 200)
    # new_network = DQN((120, 120), maze_size)
    # new_network.play_game_static(maze1, 100, "model_weights.h5")

# Static maze, no heuristics
def TestCase1():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_size = 4
    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    network = DQN((120, 120), maze_size)
    network.train_agent_static(maze1, 200)
    new_network = DQN((120, 120), maze_size)
    new_network.play_game_static(maze1, 100, "model_weights.h5")

    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8


# Static maze, has heuristics
def TestCase2():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_size = 4
    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    network = DQN((120, 120), maze_size)
    network.train_agent_static(maze1, 200, heuristics_flag=True)
    new_network = DQN((120, 120), maze_size)
    new_network.play_game_static(maze1, 100, "model_weights.h5")


# Eight different mazes
def TestCase3():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different start pt and start orientation than 1
    maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different end pt than 1
    maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_size = 4
    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180, slip_flag=True)
    # maze1.show()
    maze2 = Maze(maze_array2, marker_filepath, (1,0), (3,3), 270, slip_flag=True)
    # maze2.show()
    maze3 = Maze(maze_array3, marker_filepath, (3,3), (3,0), 180, slip_flag=True)
    # maze3.show()
    maze4 = Maze(maze_array4, marker_filepath, (0,0), (3,3), 180, slip_flag=True)
    # maze4.show()
    maze5 = Maze(maze_array5, marker_filepath, (0,0), (0,3), 180, slip_flag=True)
    # maze5.show()
    maze6 = Maze(maze_array6, marker_filepath, (3,3), (0,0), 90, slip_flag=True)
    # maze6.show()
    maze7 = Maze(maze_array7, marker_filepath, (3,3), (1,1), 90, slip_flag=True)
    # maze7.show()
    maze8 = Maze(maze_array8, marker_filepath, (3,3), (0,0), 90, slip_flag=True)
    # maze8.show()
    network = DQN((120, 120), maze_size)
    # network.train_agent_static(maze1, 200)
    network.train_agent_dynamic([maze1, maze2, maze3, maze4, maze5, maze6, maze7, maze8], 1500, heuristics_flag=True)

    # answer = input("Ready to play the game? y/n: ")
    # Create a new object, load weights, and see if it works?
    # if answer == "y":
    new_network = DQN((120, 120), maze_size)
    # new_network.play_game_static(maze1, 100, "model_weights.h5")
    new_network.play_game_dynamic([maze1, maze2, maze3, maze4, maze5, maze6, maze7, maze8], 200, "model_weights.h5")
    # if answer == "n":
    #     print("Program Exited.")

# Static maze, has heuristics
def TestCase4():
    # Using a 4x4 maze:
    # maze_array1 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # maze_size = 4
    # Using a 8x8 maze:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    maze_size = 10

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (9,9), 180, slip_flag=True)
    # maze1.show()
    network = DQN((120, 120), maze_size)
    network.train_agent_static(maze1, 200, heuristics_flag=True)
    new_network = DQN((120, 120), maze_size)
    new_network.play_game_static(maze1, 100, "model_weights.h5")
    
def TestCase5():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different start pt and start orientation than 1
    maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different end pt than 1
    maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    # #Different maze 5 not blocking best path
    # new_maze_array5 = np.array(
    # [[0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # Different maze 5 blocking best path
    new_maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_size = 4
    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    # maze1.show()
    new_maze1 = Maze(new_maze_array1, marker_filepath, (0,0), (3,3), 180)
    # new_maze1.show()
    maze2 = Maze(maze_array2, marker_filepath, (1,0), (3,3), 270)
    # maze2.show()
    new_maze2 = Maze(new_maze_array2, marker_filepath, (1,0), (3,3), 270)
    # new_maze2.show()
    maze3 = Maze(maze_array3, marker_filepath, (3,3), (3,0), 180)
    # maze3.show()
    new_maze3 = Maze(new_maze_array3, marker_filepath, (3,3), (3,0), 180)
    # new_maze3.show()
    maze4 = Maze(maze_array4, marker_filepath, (0,0), (3,3), 180)
    # maze4.show()
    new_maze4 = Maze(new_maze_array4, marker_filepath, (0,0), (3,3), 180)
    # new_maze4.show()
    maze5 = Maze(maze_array5, marker_filepath, (0,0), (0,3), 180)
    # maze5.show()
    new_maze5 = Maze(new_maze_array5, marker_filepath, (0,0), (0,3), 180)
    # new_maze5.show()
    maze6 = Maze(maze_array6, marker_filepath, (3,3), (0,0), 90)
    # maze6.show()
    new_maze6 = Maze(new_maze_array6, marker_filepath, (3,3), (0,0), 90)
    # new_maze6.show()
    maze7 = Maze(maze_array7, marker_filepath, (3,3), (1,1), 90)
    # maze7.show()
    new_maze7 = Maze(new_maze_array7, marker_filepath, (3,3), (1,1), 90)
    # new_maze7.show()
    maze8 = Maze(maze_array8, marker_filepath, (3,3), (0,0), 90)
    # maze8.show()
    new_maze8 = Maze(new_maze_array8, marker_filepath, (3,3), (0,0), 90)
    # new_maze8.show()

    # answer = input("Ready to play the game? y/n: ")
    # Create a new object, load weights, and see if it works?
    # if answer == "y":
    new_network = DQN((120, 120), maze_size)
    # new_network.play_game_static(maze1, 100, "model_weights.h5")
    new_network.play_game_dynamic([new_maze1, new_maze2, new_maze3, new_maze4, new_maze5, new_maze6, new_maze7, new_maze8], 200, "run64_model_weights.h5")

def TestCase5():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different start pt and start orientation than 1
    maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different end pt than 1
    maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    # #Different maze 5 not blocking best path
    # new_maze_array5 = np.array(
    # [[0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # Different maze 5 blocking best path
    new_maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_size = 4
    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    # maze1.show()
    new_maze1 = Maze(new_maze_array1, marker_filepath, (0,0), (3,3), 180)
    # new_maze1.show()
    maze2 = Maze(maze_array2, marker_filepath, (1,0), (3,3), 270)
    # maze2.show()
    new_maze2 = Maze(new_maze_array2, marker_filepath, (1,0), (3,3), 270)
    # new_maze2.show()
    maze3 = Maze(maze_array3, marker_filepath, (3,3), (3,0), 180)
    # maze3.show()
    new_maze3 = Maze(new_maze_array3, marker_filepath, (3,3), (3,0), 180)
    # new_maze3.show()
    maze4 = Maze(maze_array4, marker_filepath, (0,0), (3,3), 180)
    # maze4.show()
    new_maze4 = Maze(new_maze_array4, marker_filepath, (0,0), (3,3), 180)
    # new_maze4.show()
    maze5 = Maze(maze_array5, marker_filepath, (0,0), (0,3), 180)
    # maze5.show()
    new_maze5 = Maze(new_maze_array5, marker_filepath, (0,0), (0,3), 180)
    # new_maze5.show()
    maze6 = Maze(maze_array6, marker_filepath, (3,3), (0,0), 90)
    # maze6.show()
    new_maze6 = Maze(new_maze_array6, marker_filepath, (3,3), (0,0), 90)
    # new_maze6.show()
    maze7 = Maze(maze_array7, marker_filepath, (3,3), (1,1), 90)
    # maze7.show()
    new_maze7 = Maze(new_maze_array7, marker_filepath, (3,3), (1,1), 90)
    # new_maze7.show()
    maze8 = Maze(maze_array8, marker_filepath, (3,3), (0,0), 90)
    # maze8.show()
    new_maze8 = Maze(new_maze_array8, marker_filepath, (3,3), (0,0), 90)
    # new_maze8.show()

    # answer = input("Ready to play the game? y/n: ")
    # Create a new object, load weights, and see if it works?
    # if answer == "y":
    new_network = DQN((120, 120), maze_size)
    # new_network.play_game_static(maze1, 100, "model_weights.h5")
    new_network.play_game_dynamic([new_maze1, new_maze2, new_maze3, new_maze4, new_maze5, new_maze6, new_maze7, new_maze8], 200, "run64_model_weights.h5")

def TestCase6():
    # Using a 4x4 maze:
    # Testing dynamic mazes:
    maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array1 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different start pt and start orientation than 1
    maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array2 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    # Different end pt than 1
    maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array3 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array4 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    # #Different maze 5 not blocking best path
    # new_maze_array5 = np.array(
    # [[0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # Different maze 5 blocking best path
    new_maze_array5 = np.array(
    [[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    new_maze_array6 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])
    maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array7 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    new_maze_array8 = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]])
    maze_size = 4
    # # Using a 8x8 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    t = 0
    # maze1.show()
    new_maze1 = Maze(new_maze_array1, marker_filepath, (0,0), (3,3), 180)
    maze1.generate_img(t)
    t+=1
    new_maze1.generate_img(t)
    t+=1
    # new_maze1.show()
    maze2 = Maze(maze_array2, marker_filepath, (1,0), (3,3), 270)
    # maze2.show()
    new_maze2 = Maze(new_maze_array2, marker_filepath, (1,0), (3,3), 270)
    # new_maze2.show()
    maze2.generate_img(t)
    t+=1
    new_maze2.generate_img(t)
    t+=1
    maze3 = Maze(maze_array3, marker_filepath, (3,3), (3,0), 180)
    # maze3.show()
    new_maze3 = Maze(new_maze_array3, marker_filepath, (3,3), (3,0), 180)
    maze3.generate_img(t)
    t+=1
    new_maze3.generate_img(t)
    t+=1
    # new_maze3.show()
    maze4 = Maze(maze_array4, marker_filepath, (0,0), (3,3), 180)
    # maze4.show()
    new_maze4 = Maze(new_maze_array4, marker_filepath, (0,0), (3,3), 180)
    # new_maze4.show()
    maze4.generate_img(t)
    t+=1
    new_maze4.generate_img(t)
    t+=1
    maze5 = Maze(maze_array5, marker_filepath, (0,0), (0,3), 180)
    # maze5.show()
    new_maze5 = Maze(new_maze_array5, marker_filepath, (0,0), (0,3), 180)
    # new_maze5.show()
    maze5.generate_img(t)
    t+=1
    new_maze5.generate_img(t)
    t+=1
    maze6 = Maze(maze_array6, marker_filepath, (3,3), (0,0), 90)
    # maze6.show()
    new_maze6 = Maze(new_maze_array6, marker_filepath, (3,3), (0,0), 90)
    maze6.generate_img(t)
    t+=1
    new_maze6.generate_img(t)
    t+=1
    # new_maze6.show()
    maze7 = Maze(maze_array7, marker_filepath, (3,3), (1,1), 90)
    # maze7.show()
    new_maze7 = Maze(new_maze_array7, marker_filepath, (3,3), (1,1), 90)
    maze7.generate_img(t)
    t+=1
    new_maze7.generate_img(t)
    t+=1
    # new_maze7.show()
    maze8 = Maze(maze_array8, marker_filepath, (3,3), (0,0), 90)
    # maze8.show()
    new_maze8 = Maze(new_maze_array8, marker_filepath, (3,3), (0,0), 90)
    # new_maze8.show()
    maze8.generate_img(t)
    t+=1
    new_maze8.generate_img(t)
    # answer = input("Ready to play the game? y/n: ")
    # Create a new object, load weights, and see if it works?
    # if answer == "y":
    # new_network = DQN((120, 120), maze_size)
    # new_network.play_game_static(maze1, 100, "model_weights.h5")
    # new_network.play_game_dynamic([new_maze1, new_maze2, new_maze3, new_maze4, new_maze5, new_maze6, new_maze7, new_maze8], 200, "run64_model_weights.h5")

if __name__ == "__main__":
    TestCase4()
    # # Using a 4x4 maze:
    # # Testing dynamic mazes:
    # maze_array1 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # # Different start pt and start orientation than 1
    # maze_array2 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # # Different end pt than 1
    # maze_array3 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # maze_array4 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 1.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # maze_array5 = np.array(
    # [[0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # maze_array6 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # maze_array7 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # maze_array8 = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 1.0],
    # [0.0, 0.0, 0.0, 0.0]])
    # maze_size = 4
    # # # Using a 8x8 maze:
    # # maze_array = np.array(
    # # [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    # # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    # # [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    # # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    # # [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    # # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    # # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    # # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # # maze_size = 8

    # marker_filepath = "images/marker8.jpg"
    # maze1 = Maze(maze_array1, marker_filepath, (0,0), (3,3), 180)
    # # maze1.show()
    # maze2 = Maze(maze_array2, marker_filepath, (1,0), (3,3), 270)
    # # maze2.show()
    # maze3 = Maze(maze_array3, marker_filepath, (3,3), (3,0), 180)
    # # maze3.show()
    # maze4 = Maze(maze_array4, marker_filepath, (0,0), (3,3), 180)
    # # maze4.show()
    # maze5 = Maze(maze_array5, marker_filepath, (0,0), (0,3), 180)
    # # maze5.show()
    # maze6 = Maze(maze_array6, marker_filepath, (3,3), (0,0), 90)
    # # maze6.show()
    # maze7 = Maze(maze_array7, marker_filepath, (3,3), (1,1), 90)
    # # maze7.show()
    # maze8 = Maze(maze_array8, marker_filepath, (3,3), (0,0), 90)
    # # maze8.show()
    # network = DQN((120, 120), maze_size)
    # # network.train_agent_static(maze1, 200)
    # network.train_agent_dynamic([maze1, maze2, maze3, maze4, maze5, maze6, maze7, maze8], 3000, heuristics_flag=True)

    # # answer = input("Ready to play the game? y/n: ")
    # # Create a new object, load weights, and see if it works?
    # # if answer == "y":
    # new_network = DQN((120, 120), maze_size)
    # # new_network.play_game_static(maze1, 100, "model_weights.h5")
    # new_network.play_game_dynamic([maze1, maze2, maze3, maze4, maze5, maze6, maze7, maze8], 200, "model_weights.h5")
    # # if answer == "n":
    # #     print("Program Exited.")



    # # Testing for saving and loading weights
    # original_weights = network.model.get_weights()
    # loaded_weights = new_network.model.get_weights()
    # for layer_idx in range(len(original_weights)):
    #     print("Layer", layer_idx)
    #     # print("Original Weights:", original_weights[layer_idx])
    #     # print("Loaded Weights:", loaded_weights[layer_idx])
    #     print("Weights Match:", (original_weights[layer_idx] == loaded_weights[layer_idx]).all())
    #     print()
        
    # # Testing for autotest.py
    # run = 0
    # folder_path = 'autotest_results'
    # value_lst = [0.3, 0.5, 0.6, 0.8, 0.10]
    # for value in value_lst:
    #     if os.path.isfile(folder_path + '/' + str(run) + '.png'):
    #         run += 1
    #         continue
    #     print(value)
    