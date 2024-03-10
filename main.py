import numpy as np
from class_maze import Maze
from class_DQN import DQN
import matplotlib.pyplot as plt
from itertools import count
import random
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
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
    # Testing train_agent:
    maze_array = np.array(
    [[0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]])
    marker_filepath = "images/marker8.jpg"
    maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
    network = DQN((389, 389))
    network.train_agent(maze, 35)
    rewards = network.episode_rewards_lst
    plt.plot([i for i in range(0, len(rewards))], rewards, color='blue', linestyle='-', marker='o', label='Lines')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

