import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, maze, start_pt: tuple, goal_pt: tuple):
        self.reset_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        self.traversed = []
#        self.traversed = np.array([]) # creates an empty numpy array

    def show(self):
        plt.grid(True)
        nrows, ncols = self.maze.shape
        # print(self.maze.shape)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START')
        ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL')
        self.maze[self.robot_location[0], self.robot_location[1]] = 0.7
        for x, y in self.traversed:
            self.maze[x, y] = 0.5
        img = plt.imshow(self.maze, interpolation='none', cmap='binary')
        img = plt.show()
        return img

    def reset(self):
        self.maze = self.reset_maze
        self.robot_location = self.start_pt
        # self.traversed = np.array([])
        self.traversed = []

    def move_robot(self, direction:str):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TO-DO: TAKE INTO ACCOUNT IF IT HITS A MAZE EDGE.
        # TO-DO: Consider if I still need to append to a traversed location in line above, if robot does not move from invalid move.
        if direction == "LEFT":
            test_location = (robot_x, robot_y-1)
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[0], test_location[1]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y-1)
        elif direction == "RIGHT":
            test_location = (robot_x, robot_y+1)
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[0], test_location[1]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y+1)
        elif direction == "UP":
            test_location = (robot_x-1, robot_y)
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[0], test_location[1]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x-1, robot_y)
        elif direction == "DOWN":
            test_location = (robot_x+1, robot_y)
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[0], test_location[1]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x+1, robot_y)

    # def check_move(self, direction:str):
    #     robot_x, robot_y = self.robot_location[0], self.robot_location[1]
    #     if direction == "LEFT":
    #         test_location = (robot_x, robot_y-1)
    #     elif direction == "RIGHT":
    #         test_location = (robot_x, robot_y+1)
    #     elif direction == "UP":
    #         test_location = (robot_x-1, robot_y)
    #     elif direction == "DOWN":
    #         test_location = (robot_x+1, robot_y)
    #     # If wall detected:
    #     if self.maze[test_location[0], test_location[1]] == 1:
    #         print ("ERROR: Wall detected. Cannot traverse.")
    #         return
    #     # TO-DO: Consider what if robot exits the maze?
    #     # TO-DO: If robot stays in place, do I add that to the traversed list too?
    #     else:
    #         if direction == "LEFT":
    #             return 4
    #         elif direction == "RIGHT":
    #             return 2
    #         elif direction == "UP":
    #             return 1
    #         elif direction == "DOWN":
    #             return 3

    def produce_video():
        pass