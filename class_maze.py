import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, maze, start_pt: tuple, goal_pt: tuple):
        self.maze = maze
        self.robot_location = start_pt
        self.start_pt = start_pt
        self.goal_pt = goal_pt

    def show(self):
        plt.grid(True)
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START')
        ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL')
        self.maze[self.robot_location[0], self.robot_location[1]] = 0.7
#        self.maze[self.start_pt[0], self.start_pt[1]] = 0.6
#        maze[self.end_pt[0], self.end_pt[1]] = 0.3
        # maze[2,1] = 0.6
        # canvas = np.copy(maze)
    #    for row,col in maze.visited:
    #        canvas[row,col] = 0.6
    #    rat_row, rat_col, _ = maze.state

        img = plt.imshow(self.maze, interpolation='none', cmap='binary')
        img = plt.show()
        return img

    def reset(self, maze):
        pass

    def move_robot(self, direction:str):
        pass
        # if move is valid, if not, then:
        if direction == "LEFT":
            self.robot_location = (self.robot_location[0], self.robot_location[1]-1)
        elif direction == "RIGHT":
            self.robot_location = (self.robot_location[0], self.robot_location[1]+1)
        elif direction == "UP":
            self.robot_location = (self.robot_location[0]-1, self.robot_location[1])
        elif direction == "DOWN":
            self.robot_location = (self.robot_location[0]+1, self.robot_location[1])


# Do we really need this if we have move_robot?
    def update(maze):
        pass

    def produce_video():
        pass