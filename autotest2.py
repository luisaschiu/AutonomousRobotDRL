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

if __name__ == "__main__":
    # Test varying goal reward with larger final exploration frame
    goal = 1
    visited = -0.25
    new_step = -0.04
    run = 0
    lst = []
    maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0]])
    marker_filepath = "images/marker8.jpg"
    # Test changing visited reward:
    folder_path = 'autotest_results'
    for i in range(100, 10000, 10):
        lst.append(i/100)
    for value in lst:
        if os.path.isfile(folder_path + 'expl_rate_/' + str(run) + '.png'):
            run += 1
            continue
        goal = value
        maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 180)
        network = DQN(state_size = (120, 120))
        network.train_agent(maze, 100, goal_rwd = goal, visited_rwd= visited, new_step_rwd = new_step)
        rewards_lst = network.episode_rewards_lst
        total_step_loss_lst = network.total_step_loss_lst
        loss_lst = network.loss_lst
        expl_rate_lst = network.expl_rate_lst
        # Plot and save rewards info
        # plt.figure(2)
        plt.clf()
        plt.plot([i for i in range(0, len(rewards_lst))], rewards_lst, color='blue', linestyle='-', marker='o', label='Lines')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('REWARDS: goal = ' + str(goal) + ', visited = ' + str(visited) + ', new_step = ' + str(new_step))
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        # Save the plot to the folder
        plt.savefig(os.path.join(folder_path + '/' + 'rewards_' + str(run) + '.png'))
        plt.clf()

        # Plot and save loss info
        # plt.figure(3)
        plt.plot(total_step_loss_lst, loss_lst, color='blue', linestyle='-', marker='o', label='Lines')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('REWARDS: goal = ' + str(goal) + ', visited = ' + str(visited) + ', new_step = ' + str(new_step))
        # Save the plot to the folder
        plt.savefig(os.path.join(folder_path + '/' + 'loss_' + str(run) + '.png'))
        plt.clf()

        # Plot and save expl_rate info
        # plt.figure(4)
        plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst, color='blue', linestyle='-', marker='o', label='Lines')
        plt.xlabel('steps')
        plt.ylabel('exploration rate')
        plt.title('REWARDS: goal = ' + str(goal) + ', visited = ' + str(visited) + ', new_step = ' + str(new_step))
        # Save the plot to the folder
        plt.savefig(os.path.join(folder_path + '/' + 'expl_rate_' + str(run) + '.png'))
        plt.clf()

        run +=1
        print("Run: ", run)