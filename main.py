from class_maze import Maze
from class_DQN import DQN
from Maze_Image_Generation import generateMaze, prepareGrid,rgb_to_bw, remove_border, longest_path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_errors(min_steps, filename):
    # Read the csv file
    data = pd.read_csv(filename)
    
    # Ensure the 'Steps' column exists
    if 'Steps' not in data.columns:
        print("The 'Steps' column does not exist in the provided CSV file.")
        return

    # Filter the data
    filtered_data = data[data['Steps'] >= min_steps]

    # Calculate the errors
    mae = mean_absolute_error(filtered_data['Steps'], [min_steps]*len(filtered_data))
    mse = mean_squared_error(filtered_data['Steps'], [min_steps]*len(filtered_data))

    # Print the errors
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":

    dim = int(input('Enter a maze size: '))
    episodes = int(input('Enter number of episodes: '))
    directory = f'results_{dim}x{dim}_for_{episodes}'


    grid = generateMaze(dim)
    gridRGB = prepareGrid(grid)
    maze_array = rgb_to_bw(gridRGB)
    maze_array = remove_border(maze_array)
    start, goal, min_steps = longest_path(maze_array)

    start = (start[1], start[0])
    goal = (goal[1], goal[0])
    if dim == 4:
        start = (0,0)
        goal = (3,3)
        min_steps = 6
        # Using a 4x4 maze:
        maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0]])

    if dim == 8:
        start = (0,0)
        goal = (7,7)
        min_steps = 14
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

    print(f"Maze start at position: {start}")
    print(f"Maze goal at position: {goal}")
    marker_filepath = "images/marker8.jpg"
    goal_filepath = "images/star.jpg"
    maze = Maze(maze_array, marker_filepath, goal_filepath, start, goal, 180, hidden_goal=True)
    init_state = maze.reset(0)
    network = DQN((init_state.shape), len(maze_array[0]),result_directory=directory,result_filepath='train_episodes')
    network.train_agent(maze,episodes)

    rewards_lst = network.episode_rewards_lst
    total_step_loss_lst = network.total_step_loss_lst
    loss_lst = network.loss_lst
    expl_rate_lst = network.expl_rate_lst
    
    data_folder_path = f'{directory}/data_plots'
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

    new_network = DQN((init_state.shape), len(maze_array[0]),result_directory=directory,result_filepath='test_episodes')
    new_network.play_game(maze, 100, f"{directory}/model_weights.h5")
    calculate_errors(min_steps,f'{directory}/gameplay_data.csv')

