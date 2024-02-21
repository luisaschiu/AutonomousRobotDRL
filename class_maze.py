import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import ArUco_functions
import os
import glob

class Maze:
    def __init__(self, maze, marker_filepath, start_pt: tuple, goal_pt: tuple, start_orientation):
        self.init_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.start_orientation = start_orientation
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        self.traversed = []
        self.min_reward = -0.5*maze.size
        self.total_reward = 0
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

    def generate_img(self, time_step):
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
        fig = plt.savefig('robot_steps/' + str(time_step) + '.jpg', bbox_inches='tight')
        # fig = plt.savefig('robot_steps/' + str(self.time_step) + '.jpg', bbox_inches=Bbox.from_bounds(1, 1, 4, 4))
        plt.close(fig)
        image = cv.imread('robot_steps/' + str(time_step) + '.jpg')
        # cv.imshow('img', image)
        # cv.waitKey(0)
        return image
        

    def reset(self, time_step):
        self.maze = self.init_maze
        self.robot_location = self.start_pt
        # print("Robot_orientation: ", self.robot_orientation)
        # print("Start_orientation: ", self.start_orientation)
        self.robot_orientation = self.start_orientation//90
        # self.traversed = np.array([])
        # Reset previously traversed locations for the next episode
        self.traversed = []
        self.timestep = 0
        self.total_reward = 0
        cur_state_img = self.generate_img(time_step)
        return cur_state_img

    def move_robot(self, direction:str):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TODO: TAKE INTO ACCOUNT IF IT HITS A MAZE EDGE?
        # TODO: Consider if I still need to append to a traversed location in line above, if robot does not move from invalid move.
        if direction == "UP":
            test_location = (robot_x, robot_y-1)
            expected_angle = 0
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y-1)
        elif direction == "DOWN":
            test_location = (robot_x, robot_y+1)
            expected_angle = 180
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y+1)
        elif direction == "LEFT":
            test_location = (robot_x-1, robot_y)
            expected_angle = 90
            # Maze Edge Check
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x-1, robot_y)
        elif direction == "RIGHT":
            test_location = (robot_x+1, robot_y)
            expected_angle = 270
            # Maze Edge Check
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise CustomError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x+1, robot_y)
                
    def get_available_actions(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        valid_actions = []
        invalid_actions = [] #NOTE: For testing purposes, could just check for conditions rather than do the if/elif/else statements
        # valid_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Testing UP action:
        up_location = (robot_x, robot_y-1)
        # Maze Edge Check
        if ((up_location[0]) < 0 or (up_location[1] < 0) or (up_location[0] > self.maze.shape[0]-1) or (up_location[1] > self.maze.shape[1]-1)):
            invalid_actions.append("UP")
        # Wall Check
        elif self.maze[up_location[1], up_location[0]] == 1:
            invalid_actions.append("UP")
        else:
            valid_actions.append("UP")

        # Testing DOWN action:
        down_location = (robot_x, robot_y+1)
        # Maze Edge Check
        if ((down_location[0]) < 0 or (down_location[1] < 0) or (down_location[0] > self.maze.shape[0]-1) or (down_location[1] > self.maze.shape[1]-1)):
            invalid_actions.append("DOWN")
        # Wall Check
        elif self.maze[down_location[1], down_location[0]] == 1:
            invalid_actions.append("DOWN")
        else:
            valid_actions.append("DOWN")
        
        # Testing LEFT action:
        left_location = (robot_x-1, robot_y)
        # Maze Edge Check
        if ((left_location[1]) < 0 or (left_location[0] < 0) or (left_location[0] > self.maze.shape[0]-1) or (left_location[1] > self.maze.shape[1]-1)):
            invalid_actions.append("LEFT")
        # Wall Check
        elif self.maze[left_location[1], left_location[0]] == 1:
            invalid_actions.append("LEFT")
        else:
            valid_actions.append("LEFT")

        # Testing RIGHT action:
        right_location = (robot_x+1, robot_y)
        # Maze Edge Check
        if ((right_location[1]) < 0 or (right_location[0] < 0) or (right_location[0] > self.maze.shape[0]-1) or (right_location[1] > self.maze.shape[1]-1)):
            invalid_actions.append("RIGHT")
        # Wall Check
        elif self.maze[right_location[1], right_location[0]] == 1:
            invalid_actions.append("RIGHT")
        else:
            valid_actions.append("RIGHT")

        if len(valid_actions) == 0:
            print("ERROR: Robot cannot move!")
        return valid_actions

    def get_reward(self):
        # NOTE: Do I account for maze edges or walls here?
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # Robot reached the goal
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return 1
        # Robot has already visited this spot
        if (robot_x, robot_y) in self.traversed:
            return -0.25
        else:
            # Advanced onto a new spot in the maze, but hasn't reached the goal or gone backwards
            return -1
    
    def game_over(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # If rewards value is less than the minimum rewards allowed
        if self.total_reward < self.min_reward:
            return True
        # If goal is reached
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return True
        return False
        # if self.total_reward < self.min_reward:
        #     return 'lose'
        # # If goal is reached
        # if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
        #     return 'win'
        # return 'not over'

    def take_action(self, action: str, time_step):
        self.move_robot(action)
        reward = self.get_reward()
        self.total_reward += reward
        game_over = self.game_over()
        # self.time_step += 1
        new_state_img = self.generate_img(time_step)
        return (new_state_img, reward, game_over)

    def produce_video():
        pass

class CustomError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)