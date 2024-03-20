import numpy as np
from class_maze import Maze
from class_DQN import DQN
import matplotlib.pyplot as plt
from itertools import count
import random
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
import os
import threading
import time

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

def generate_data():
    episode = int(input("Input test episode num: "))
    episode_reward = float(input("Input test reward value: "))
    if episode == 0:
        save_to_csv([episode, episode_reward], "data.csv", ["Episode", "Reward"])
    else:
        save_to_csv([episode, episode_reward], "data.csv", None)

def animate(i):
    data = pd.read_csv('data.csv')
    x = data['Episode']
    y = data['Reward']
    plt.cla()
    # plt.plot(x, y)
    plt.plot(x, y, color='blue', linestyle='-', marker='o', label='Lines')


def plot_thread():
    fig = plt.figure()  # Create a new figure
    ani = FuncAnimation(fig, animate)  # Create the animation
    plt.show()  # Show the plot and animation

def data_thread():
    while True:
        generate_data()

# if __name__ == "__main__":
#     plot_thread = threading.Thread(target=plot_thread, daemon = True)
#     plot_thread.start()

#     data_thread = threading.Thread(target=data_thread, daemon = True)
#     data_thread.start()

#     while True:
#         time.sleep(1)  # Keep the main thread running
        
if __name__ == "__main__":
    # Using a 4x4 maze:
    # maze_array = np.array(
    # [[0.0, 1.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0, 0.0]])
    # maze_size = 4
    # Using a 8x8 maze:
    maze_array = np.array(
    [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    maze_size = 8

    marker_filepath = "images/marker8.jpg"
    maze = Maze(maze_array, marker_filepath, (0,0), (7,7), 180)
    network = DQN((120, 120), maze_size)
    network.train_agent(maze, 35)
    rewards_lst = network.episode_rewards_lst
    total_step_loss_lst = network.total_step_loss_lst
    loss_lst = network.loss_lst
    expl_rate_lst = network.expl_rate_lst
    
    data_folder_path = 'data_plots'
    os.makedirs(data_folder_path, exist_ok = True)
    # plt.clf()
    # plt.plot([i for i in range(0, len(rewards_lst))], rewards_lst, color='blue', linestyle='-', label='Lines')
    plt.plot([i for i in range(0, len(rewards_lst))], rewards_lst)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig(os.path.join(data_folder_path + '/rewards.png'))
    plt.clf()

    # plt.plot(total_step_loss_lst, loss_lst, color='blue', linestyle='-', label='Lines')
    plt.plot(total_step_loss_lst, loss_lst)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(data_folder_path + '/loss.png'))
    plt.clf()

    # plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst, color='blue', linestyle='-', label='Lines')
    plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst)
    plt.xlabel('Steps')
    plt.ylabel('Expl_rate')
    plt.savefig(os.path.join(data_folder_path + '/expl_rate.png'))
    plt.clf()

    answer = input("Ready to play the game? y/n: ")
    # Create a new object, load weights, and see if it works?
    if answer == "y":
        new_network = DQN((120, 120), maze_size)
        new_network.play_game(maze, 100, "model_weights.h5")
    if answer == "n":
        print("Program Exited.")


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
    