import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
import AruCo_functions
import os
import glob

class Maze:
    def __init__(self, maze:np.array, marker_filepath:str, goal_filepath:str, start_pt: tuple, goal_pt: tuple, start_orientation:int, hidden_goal=True):
        self.init_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.init_orientation = start_orientation
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        self.goal = mpimg.imread(goal_filepath)
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        self.traversed = []
        self.num_traversed = 0
        # self.min_reward = -2*maze.size
        self.total_reward = 0
        self.hidden_goal = hidden_goal
        self.init_shape = None

    def generate_img(self, time_step):
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        scaled_zoom =(1/nrows)

        marker = self.marker
        # Rotate the marker around its center
        marker = rotate(marker, self.robot_orientation * 90, reshape=False)
        imagebox = OffsetImage(marker, zoom=scaled_zoom, cmap='gray')
        ab = AnnotationBbox(imagebox, (self.robot_location[0], self.robot_location[1]), frameon=False)
        ax.add_artist(ab)

        if self.hidden_goal is False:
            goal = self.goal
            imagebox = OffsetImage(goal, zoom=scaled_zoom, cmap='winter')
            ab = AnnotationBbox(imagebox, (self.goal_pt[0], self.goal_pt[1]), frameon=False)
            ax.add_artist(ab)
        
        directory = 'robot_steps/'
        plt.imshow(self.maze, interpolation='none', cmap='binary')
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig = plt.savefig(directory + str(time_step) + '.jpg', bbox_inches='tight')
        plt.close(fig)
        
        # Read the image and ensure it is square
        image = cv.imread(directory + str(time_step) + '.jpg')
        if self.init_shape is None:
            height, width = image.shape[:2]
            if height != width:
                size = min(height, width)
                image = cv.resize(image, (size, size))
        else:
            image = cv.resize(image, (self.init_shape[1], self.init_shape[0]))
        
        cv.imwrite(directory + str(time_step) + '.jpg', image)
        cv.imshow('Frame', image)
        cv.waitKey(1)
        # print(image.shape)
        return image

            

    def reset(self, time_step):
        for filename in glob.glob('robot_steps/*.jpg'):
            os.remove(filename)
        self.maze = self.init_maze
        self.robot_location = self.start_pt
        self.robot_orientation = self.init_orientation//90
        self.traversed = []
        self.timestep = 0
        self.total_reward = 0

        directory = 'robot_steps/'
        img_files = glob.glob(os.path.join(directory, "*.[pjJ][npNP][gG]*"))
        for img_file in img_files:
           os.remove(img_file)

        cur_state_img = self.generate_img(time_step)
        self.init_shape = cur_state_img.shape
        return cur_state_img

    def move_robot(self, direction:str):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        if direction == "UP":
            test_location = (robot_x, robot_y-1)
            expected_angle = 0
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            elif self.maze[test_location[1], test_location[0]] == 1:
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y-1)
        elif direction == "DOWN":
            test_location = (robot_x, robot_y+1)
            expected_angle = 180
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            elif self.maze[test_location[1], test_location[0]] == 1:
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x, robot_y+1)
        elif direction == "LEFT":
            test_location = (robot_x-1, robot_y)
            expected_angle = 90
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            elif self.maze[test_location[1], test_location[0]] == 1:
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x-1, robot_y)
        elif direction == "RIGHT":
            test_location = (robot_x+1, robot_y)
            expected_angle = 270
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            elif self.maze[test_location[1], test_location[0]] == 1:
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x+1, robot_y)
                
    def get_available_actions(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        valid_actions = []

        # Testing UP action:
        up_location = (robot_x, robot_y-1)
        # Maze Edge Check
        if ((up_location[0]) < 0 or (up_location[1] < 0) or (up_location[0] > self.maze.shape[0]-1) or (up_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(0)
        # Wall Check
        elif self.maze[up_location[1], up_location[0]] == 1:
            valid_actions.append(0)
        else:
            valid_actions.append(1)

        # Testing DOWN action:
        down_location = (robot_x, robot_y+1)
        # Maze Edge Check
        if ((down_location[0]) < 0 or (down_location[1] < 0) or (down_location[0] > self.maze.shape[0]-1) or (down_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(0)
        # Wall Check
        elif self.maze[down_location[1], down_location[0]] == 1:
            valid_actions.append(0)
        else:
            valid_actions.append(1)
        
        # Testing LEFT action:
        left_location = (robot_x-1, robot_y)
        # Maze Edge Check
        if ((left_location[1]) < 0 or (left_location[0] < 0) or (left_location[0] > self.maze.shape[0]-1) or (left_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(0)
        # Wall Check
        elif self.maze[left_location[1], left_location[0]] == 1:
            valid_actions.append(0)
        else:
            valid_actions.append(1)

        # Testing RIGHT action:
        right_location = (robot_x+1, robot_y)
        # Maze Edge Check
        if ((right_location[1]) < 0 or (right_location[0] < 0) or (right_location[0] > self.maze.shape[0]-1) or (right_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(0)
        # Wall Check
        elif self.maze[right_location[1], right_location[0]] == 1:
            valid_actions.append(0)
        else:
            valid_actions.append(1)

        if all(action is None for action in valid_actions):
            raise ActionError("No actions available. Robot cannot move!")
        return valid_actions

    def get_reward(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # Robot reached the goal
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            self.num_traversed = 0
            return 1
        # Robot has already visited this spot
        if (robot_x, robot_y) in self.traversed:
            self.num_traversed = self.num_traversed + 1
            return -0.04 * self.num_traversed
        else:
            # Advanced onto a new spot in the maze, but hasn't reached the goal or gone backwards
            return -0.04
    
    def game_over(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TODO: Get rid of this, account for maximum time steps allowed in class DQN
        # If rewards value is less than the minimum rewards allowed
        # if self.total_reward < self.min_reward:
        #     return True
        # If goal is reached
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return True
        return False

    def take_action(self, action: str, time_step):
        self.move_robot(action)
        self.total_reward += self.get_reward()
        print(self.total_reward)
        return (self.generate_img(time_step), self.get_reward(), self.game_over())

    def produce_video():
        pass

class ActionError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)