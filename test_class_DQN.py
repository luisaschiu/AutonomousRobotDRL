from class_maze import Maze
from class_DQN import DQN
from Maze_Image_Generation import generateMaze, prepareGrid,rgb_to_bw, remove_border, longest_path


if __name__ == "__main__":

    dim = int(input('Enter a maze size: '))

    grid = generateMaze(dim)
    gridRGB = prepareGrid(grid)
    maze_array = rgb_to_bw(gridRGB)
    maze_array = remove_border(maze_array)
    start, goal = longest_path(maze_array)

    print(maze_array)
    print(maze_array.shape)
    #TODO: Switch coords (x,y) to (y,x)
    start = (start[1], start[0])
    goal = (goal[1], goal[0])
    print(f"Maze start at position: {start}")
    print(f"Maze goal at position: {goal}")
    marker_filepath = "images/marker8.jpg"
    goal_filepath = "images/star.jpg"
    maze = Maze(maze_array, marker_filepath, goal_filepath, start, goal, 180)
    init_state = maze.reset(0)
    print(init_state.shape)
    network = DQN((init_state.shape))
    history = network.train_agent(maze,100)