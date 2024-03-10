from class_maze import Maze
from class_DQN import DQN
from Maze_Image_Generation import generateMaze, prepareGrid,rgb_to_bw, remove_border, longest_path
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    dim = int(input('Enter a maze size: '))

    grid = generateMaze(dim)
    gridRGB = prepareGrid(grid)
    maze_array = rgb_to_bw(gridRGB)
    maze_array = remove_border(maze_array)
    start, goal = longest_path(maze_array)

    start = (start[1], start[0])
    goal = (goal[1], goal[0])
    print(f"Maze start at position: {start}")
    print(f"Maze goal at position: {goal}")
    marker_filepath = "images/marker8.jpg"
    goal_filepath = "images/star.jpg"
    maze = Maze(maze_array, marker_filepath, goal_filepath, start, goal, 180, hidden_goal=True)
    init_state = maze.reset(0)
    network = DQN((init_state.shape), len(maze_array[0]))
    history = network.train_agent(maze,20)

    # Unpack the tuples into two lists
    episodes, scores = zip(*history)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Episode Scores Over Time')
    plt.grid(True)

    # Create the directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the figure to the directory
    plt.savefig('results/episode_scores.png')

    # Close the plot
    plt.close()