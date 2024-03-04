import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import AruCo_functions
import os
import glob

class Maze:
    def __init__(self, maze:np.array, marker_filepath:str, goal_filepath:str, start_pt: tuple, goal_pt: tuple, start_orientation:int):
        self.init_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.init_orientation = start_orientation
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        self.goal_pc = mpimg.imread(goal_filepath)
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        # NOTE: Might not need self.traversed anymore, since class_DQN is taking care of the history/memorizing episodes
        self.traversed = []
        self.min_reward = -2*maze.size
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

        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation) # k = 1 means rotate it 90 degrees CC
        imagebox = OffsetImage(marker, zoom = 0.20, cmap = 'gray')

        ab = AnnotationBbox(imagebox, (self.robot_location[0], self.robot_location[1]), frameon = False)
        ax.add_artist(ab)

        img = plt.imshow(self.maze, interpolation='none', cmap='binary')
        plt.show()
        return img

    def resize_image_to_square(self, image_path, target_size):
        """
        Resizes an image to a square shape with the specified target size.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the resized image.
            target_size (int): Desired size for both width and height.

        Returns:
            None
        """
        # Read the input image
        image = cv.imread(image_path)

        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the scaling factor to make the image square
        scale_factor = target_size / max(original_height, original_width)

        # Resize the image to the target size
        resized_image = cv.resize(image, (int(original_width * scale_factor), int(original_height * scale_factor)))

        # Save the resized image
        cv.imwrite(image_path, resized_image)

    def generate_img(self, time_step):
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        scaled_zoom =1/nrows

        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation)  # k = 1 means rotate it 90 degrees CC
        # starbox = OffsetImage(self.goal_pc, zoom=scaled_zoom)
        # goal = AnnotationBbox(starbox, (self.goal_pt[0], self.goal_pt[1]), frameon=False)
        # ax.add_artist(goal)
        imagebox = OffsetImage(marker, zoom=scaled_zoom, cmap='gray')
        ab = AnnotationBbox(imagebox, (self.robot_location[0], self.robot_location[1]), frameon=False)
        ax.add_artist(ab)

        directory = 'robot_steps/'
        plt.imshow(self.maze, interpolation='none', cmap='binary')
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig = plt.savefig(directory + str(time_step) + '.jpg', bbox_inches='tight')
        plt.close(fig)
        image = cv.imread(directory + str(time_step) + '.jpg')
        # self.resize_image_to_square(directory + str(time_step) + '.jpg', image.shape[0])

        cv.imshow('Frame', image)
        cv.waitKey(50)
        return image

            

    def reset(self, time_step):
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
        return cur_state_img

    def move_robot(self, direction:str):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TODO: Consider if I still need to append to a traversed location in line above, if robot does not move from invalid move.
        if direction == "UP":
            test_location = (robot_x, robot_y-1)
            expected_angle = 0
            # Maze Edge Check
            if ((test_location[0]) < 0 or (test_location[1] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
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
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
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
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x-1, robot_y)
        elif direction == "RIGHT":
            test_location = (robot_x+1, robot_y)
            expected_angle = 270
            # Maze Edge Check
            if ((test_location[1]) < 0 or (test_location[0] < 0) or (test_location[0] > self.maze.shape[0]-1) or (test_location[1] > self.maze.shape[1]-1)):
                # print ("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Maze Edge detected. Cannot traverse " + direction + ".")
            # Wall Check
            elif self.maze[test_location[1], test_location[0]] == 1:
                # print ("ERROR: Wall detected. Cannot traverse " + direction + ".")
                raise ActionError("ERROR: Wall detected. Cannot traverse " + direction + ".")
            else:
                if (self.robot_orientation) != (expected_angle//90):
                    # print("Rotating Robot")
                    self.robot_orientation = expected_angle//90
                self.traversed.append((robot_x, robot_y))
                self.robot_location = (robot_x+1, robot_y)
                
    def get_available_actions(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        valid_actions = []
        # valid_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Testing UP action:
        up_location = (robot_x, robot_y-1)
        # Maze Edge Check
        if ((up_location[0]) < 0 or (up_location[1] < 0) or (up_location[0] > self.maze.shape[0]-1) or (up_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(None)
        # Wall Check
        elif self.maze[up_location[1], up_location[0]] == 1:
            valid_actions.append(None)
        else:
            valid_actions.append("UP")

        # Testing DOWN action:
        down_location = (robot_x, robot_y+1)
        # Maze Edge Check
        if ((down_location[0]) < 0 or (down_location[1] < 0) or (down_location[0] > self.maze.shape[0]-1) or (down_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(None)
        # Wall Check
        elif self.maze[down_location[1], down_location[0]] == 1:
            valid_actions.append(None)
        else:
            valid_actions.append("DOWN")
        
        # Testing LEFT action:
        left_location = (robot_x-1, robot_y)
        # Maze Edge Check
        if ((left_location[1]) < 0 or (left_location[0] < 0) or (left_location[0] > self.maze.shape[0]-1) or (left_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(None)
        # Wall Check
        elif self.maze[left_location[1], left_location[0]] == 1:
            valid_actions.append(None)
        else:
            valid_actions.append("LEFT")

        # Testing RIGHT action:
        right_location = (robot_x+1, robot_y)
        # Maze Edge Check
        if ((right_location[1]) < 0 or (right_location[0] < 0) or (right_location[0] > self.maze.shape[0]-1) or (right_location[1] > self.maze.shape[1]-1)):
            valid_actions.append(None)
        # Wall Check
        elif self.maze[right_location[1], right_location[0]] == 1:
            valid_actions.append(None)
        else:
            valid_actions.append("RIGHT")

        if all(action is None for action in valid_actions):
            raise ActionError("No actions available. Robot cannot move!")
        return valid_actions

    def get_reward(self):
        # TODO: Look into penalty reward for traversed locations and advancing to new spot (maybe swap or alter)
        # NOTE: Do I account for maze edges or walls here?
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # Robot reached the goal
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return 5
        # Robot has already visited this spot
        if (robot_x, robot_y) in self.traversed:
            return -2
        else:
            # Advanced onto a new spot in the maze, but hasn't reached the goal or gone backwards
            return 1
    
    def game_over(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # TODO: Get rid of this, account for maximum time steps allowed in class DQN
        # If rewards value is less than the minimum rewards allowed
        if self.total_reward < self.min_reward:
            return True
        # If goal is reached
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return True
        return False

    def take_action(self, action: str, time_step):
        self.move_robot(action)
        self.total_reward += self.get_reward()
        return (self.generate_img(time_step), self.get_reward(), self.game_over())

    def produce_video():
        pass

class ActionError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)