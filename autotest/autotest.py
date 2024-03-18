import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import random
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
import os
import threading
import time
from class_maze_auto import Maze_AUTO
from class_DQN_auto import DQN_AUTO

if __name__ == "__main__":
    # NOTE: If you have previously saved images in autotest_results folder, the file will not change those existing images.
    # If you are testing a whole new batch of varying parameters and want a fresh batch of images, delete existing images and then run this program.
    # If you are continuing a test of the same batch of varying parameters, just run this program and it will leave the existing images alone.
    
    ''' Test varying rewards'''
    replay_start_size = 512 #(8^3)
    final_exploration_frame = 1000
    max_steps_per_episode = 64 #(8^2)

    goal = 10
    visited = -0.06
    new_step = -0.03
    run = 0
    lst = []
    maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    marker_filepath = "images/marker8.jpg"
    maze = Maze_AUTO(maze_array, marker_filepath, (0,0), (7,7), 180)
    # maze.show()
    # Test changing new_step reward:
    folder_path = 'autotest_results'
    os.makedirs(folder_path, exist_ok = True)
    for i in range(-60, -5, 1):
        lst.append(i/100)
        print(i/100)
        # print('hi')
    for value in lst:
        if os.path.isfile(folder_path + 'expl_rate_/' + str(run) + '.png'):
            run += 1
            continue
        visited = value
        maze = Maze_AUTO(maze_array, marker_filepath, (0,0), (7,7), 180)
        network = DQN_AUTO(state_size = (120, 120), replay_start_size=replay_start_size, final_exploration_frame=final_exploration_frame, max_steps_per_episode = max_steps_per_episode)
        network.train_agent(maze, 200, goal_rwd = goal, visited_rwd= visited, new_step_rwd = new_step, save_weights_dir = ('model_weights_' + str(run)))
        rewards_lst = network.episode_rewards_lst
        total_step_loss_lst = network.total_step_loss_lst
        loss_lst = network.loss_lst
        expl_rate_lst = network.expl_rate_lst
        # Plot and save rewards info
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
        plt.plot(total_step_loss_lst, loss_lst, color='blue', linestyle='-', marker='o', label='Lines')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('REWARDS: goal = ' + str(goal) + ', visited = ' + str(visited) + ', new_step = ' + str(new_step))
        # Save the plot to the folder
        plt.savefig(os.path.join(folder_path + '/' + 'loss_' + str(run) + '.png'))
        plt.clf()

        # Plot and save expl_rate info
        plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst, color='blue', linestyle='-', marker='o', label='Lines')
        plt.xlabel('steps')
        plt.ylabel('exploration rate')
        plt.title('REWARDS: goal = ' + str(goal) + ', visited = ' + str(visited) + ', new_step = ' + str(new_step))
        # Save the plot to the folder
        plt.savefig(os.path.join(folder_path + '/' + 'expl_rate_' + str(run) + '.png'))
        plt.clf()
        run +=1
        print("Run: ", run)


    ''' Test varying exploration parameters'''
    # # NOTE: You have to still set the goal, visited, new_step rewards similar to the code functionality above
    # replay_start_size = 512
    # final_exploration_frame = 1000
    # max_steps_per_episode = 64

    # goal = 10
    # visited = -0.06
    # new_step = -0.03
    # run = 0
    # lst = []
    # maze_array = np.array(
    #     [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    #     [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    #     [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # marker_filepath = "images/marker8.jpg"
    # # Test changing new_step reward:
    # folder_path = 'autotest_results'
    # # This is where you vary the parameter:
    # for i in range(1000, 5000, 500):
    #     lst.append(i)
    # for value in lst:
    #     if os.path.isfile(folder_path + 'expl_rate_/' + str(run) + '.png'):
    #         run += 1
    #         continue
    #     # Assign changing/varying parameter in the line below
    #     final_exploration_frame = value

    #     maze = Maze_AUTO(maze_array, marker_filepath, (0,0), (7,7), 180)
    #     network = DQN_AUTO(state_size = (120, 120), replay_start_size=replay_start_size, final_exploration_frame=final_exploration_frame, max_steps_per_episode = max_steps_per_episode)
    #     network.train_agent(maze, 200, goal_rwd = goal, visited_rwd= visited, new_step_rwd = new_step, save_weights_dir='/autotest_weights')
    #     rewards_lst = network.episode_rewards_lst
    #     total_step_loss_lst = network.total_step_loss_lst
    #     loss_lst = network.loss_lst
    #     expl_rate_lst = network.expl_rate_lst
    #     # Plot and save rewards info
    #     plt.clf()
    #     plt.plot([i for i in range(0, len(rewards_lst))], rewards_lst, color='blue', linestyle='-', marker='o', label='Lines')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Rewards')
    #     plt.title('EXPLORATION: replay_start_size = ' + str(replay_start_size) + ', final_expl_frame = ' + str(final_exploration_frame) + ',\n' + 'max_steps_per_episode = ' + str(max_steps_per_episode))
    #     # Create the folder if it doesn't exist
    #     os.makedirs(folder_path, exist_ok=True)
    #     # Save the plot to the folder
    #     plt.savefig(os.path.join(folder_path + '/' + 'rewards_' + str(run) + '.png'))
    #     plt.clf()

    #     # Plot and save loss info
    #     plt.plot(total_step_loss_lst, loss_lst, color='blue', linestyle='-', marker='o', label='Lines')
    #     plt.xlabel('steps')
    #     plt.ylabel('loss')
    #     plt.title('EXPLORATION: replay_start_size = ' + str(replay_start_size) + ', final_expl_frame = ' + str(final_exploration_frame) + ',\n' + 'max_steps_per_episode = ' + str(max_steps_per_episode))
    #     # Save the plot to the folder
    #     plt.savefig(os.path.join(folder_path + '/' + 'loss_' + str(run) + '.png'))
    #     plt.clf()

    #     # Plot and save expl_rate info
    #     plt.plot([i for i in range(0, len(expl_rate_lst))], expl_rate_lst, color='blue', linestyle='-', marker='o', label='Lines')
    #     plt.xlabel('steps')
    #     plt.ylabel('exploration rate')
    #     plt.title('EXPLORATION: replay_start_size = ' + str(replay_start_size) + ', final_expl_frame = ' + str(final_exploration_frame) + ',\n' + 'max_steps_per_episode = ' + str(max_steps_per_episode))
    #     # Save the plot to the folder
    #     plt.savefig(os.path.join(folder_path + '/' + 'expl_rate_' + str(run) + '.png'))
    #     plt.clf()
    #     run +=1
    #     print("Run: ", run)