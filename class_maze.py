import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import ArUco_functions
import os
import glob
# TODO: code what happens when the robot reached the goal, maybe in class_qlearn?

class Maze:
    def __init__(self, maze, marker_filepath, start_pt: tuple, goal_pt: tuple, start_orientation):
        self.reset_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        self.traversed = []
        self.time_step = 0
#        self.traversed = np.array([]) # creates an empty numpy array

    def show(self):
        # plt.grid(True)
        nrows, ncols = self.maze.shape
        # print(self.maze.shape)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START', color = 'green')
        ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL', color = 'red')
        # Overlay marker onto the robot location
        # Code from: https://towardsdatascience.com/how-to-add-an-image-to-a-matplotlib-plot-in-python-76098becaf53
        # file = "images/marker8.jpg"
        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation) # k = 1 means rotate it 90 degrees CC
        imagebox = OffsetImage(marker, zoom = 0.20, cmap = 'gray')
        # TODO: Make zoom relative to maze size above, or else changing to a 
        # larger maze may make the marker image too large compared to small maze squares
        ab = AnnotationBbox(imagebox, (self.robot_location[0], self.robot_location[1]), frameon = False)
        ax.add_artist(ab)
        # self.maze[self.robot_location[0], self.robot_location[1]] = 0.7
        # Color the traversed locations
        # for x, y in self.traversed:
        #     # NOTE: Numpy array axes are different from what I defined as the axes.
        #     self.maze[y, x] = 0.5
        img = plt.imshow(self.maze, interpolation='none', cmap='binary')
        plt.show()
        return img

    def generate_img(self):
        # plt.grid(True)
        nrows, ncols = self.maze.shape
        # print(self.maze.shape)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START', color = 'green')
        ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL', color = 'red')
        # Overlay marker onto the robot location
        # Code from: https://towardsdatascience.com/how-to-add-an-image-to-a-matplotlib-plot-in-python-76098becaf53
        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation) # k = 1 means rotate it 90 degrees CC
        imagebox = OffsetImage(marker, zoom = 0.20, cmap = 'gray')
        # TODO: Make zoom relative to maze size above, or else changing to a 
        # larger maze may make the marker image too large compared to small maze squares
        ab = AnnotationBbox(imagebox, (self.robot_location[0], self.robot_location[1]), frameon = False)
        ax.add_artist(ab)
        # self.maze[self.robot_location[0], self.robot_location[1]] = 0.7
        # Color the traversed locations
        # for x, y in self.traversed:
        #     # NOTE: Numpy array axes are different from what I defined as the axes.
        #     self.maze[y, x] = 0.5
        plt.imshow(self.maze, interpolation='none', cmap='binary')
        # Check if folder/file path exists. If not, create one.
        if not os.path.exists('robot_steps/'):
            os.makedirs('robot_steps/')
        # Save as a .jpg picture, named as current time step
        fig = plt.savefig('robot_steps/' + str(self.time_step) + '.jpg', bbox_inches='tight')
        # fig = plt.savefig('robot_steps/' + str(self.time_step) + '.jpg', bbox_inches=Bbox.from_bounds(1, 1, 4, 4))
        plt.close(fig)

    def reset(self):
        self.maze = self.reset_maze
        self.robot_location = self.start_pt
        # self.traversed = np.array([])
        # Reset previously traversed locations for the next episode
        self.traversed = []
        self.timestep = 0

    # Consider making this a separate function or class. Maybe class robot?
    def move_robot(self, direction:str):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TODO: TAKE INTO ACCOUNT IF IT HITS A MAZE EDGE?
        # TODO: Consider if I still need to append to a traversed location in line above, if robot does not move from invalid move.
        if direction == "UP":
            test_location = (robot_x, robot_y-1)
            expected_angle = 0
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y-1)
                self.time_step += 1
        elif direction == "DOWN":
            test_location = (robot_x, robot_y+1)
            expected_angle = 180
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y+1)
                self.time_step += 1
        elif direction == "LEFT":
            test_location = (robot_x-1, robot_y)
            expected_angle = 90
            # Maze Edge Check
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x-1, robot_y)
                self.time_step += 1
        elif direction == "RIGHT":
            test_location = (robot_x+1, robot_y)
            expected_angle = 270
            # Maze Edge Check
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x+1, robot_y)
                self.time_step += 1

    # def available_actions(self, direction:str):
    #     robot_x, robot_y = self.robot_location[0], self.robot_location[1]
                
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