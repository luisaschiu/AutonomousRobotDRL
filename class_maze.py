import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import glob
from PIL import Image

class Maze:
    def __init__(self, maze:np.array, marker_filepath:str, start_pt: tuple, goal_pt: tuple, start_orientation:int):
        self.init_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.init_orientation = start_orientation
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        self.start_pt = start_pt
        self.goal_pt = goal_pt
        # NOTE: Might not need self.traversed anymore, since class_DQN is taking care of the history/memorizing episodes
        self.traversed = []
        # self.min_reward = -0.5*maze.size
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
        # ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START', color = 'green')
        # ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL', color = 'red')
        self.maze[self.start_pt[0], self.start_pt[1]] = 0.3
        self.maze[self.goal_pt[0], self.goal_pt[1]] = 0.6
        # Overlay marker onto the robot location
        # Code from: https://towardsdatascience.com/how-to-add-an-image-to-a-matplotlib-plot-in-python-76098becaf53
        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation) # k = 1 means rotate it 90 degrees CC
        imagebox = OffsetImage(marker, zoom = 1/(nrows+1), cmap = 'gray')
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
        IMAGE_DIGITS = 2 # images go up to two digits, used to prepend 0's to the front of image names
        # plt.grid(True)
        nrows, ncols = self.maze.shape
        # print(self.maze.shape)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(self.start_pt[0]-0.2, self.start_pt[1]+0.05, 'START', color = 'green')
        # ax.text(self.goal_pt[0]-0.2, self.goal_pt[1]+0.05, 'GOAL', color = 'red')
        self.maze[self.start_pt[0], self.start_pt[1]] = 0.3
        self.maze[self.goal_pt[0], self.goal_pt[1]] = 0.6
        # Overlay marker onto the robot location
        # Code from: https://towardsdatascience.com/how-to-add-an-image-to-a-matplotlib-plot-in-python-76098becaf53
        marker = self.marker
        marker = np.rot90(self.marker, k=self.robot_orientation) # k = 1 means rotate it 90 degrees CC
        imagebox = OffsetImage(marker, zoom = 1/(nrows+1), cmap = 'gray')
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
        image_num = str(time_step)
        image_num = (IMAGE_DIGITS - len(image_num)) * "0" + image_num
        fig = plt.savefig('robot_steps/' + image_num + '.jpg', bbox_inches='tight')
        # fig = plt.savefig('robot_steps/' + str(self.time_step) + '.jpg', bbox_inches=Bbox.from_bounds(1, 1, 4, 4))
        plt.close(fig)
        image = cv.imread('robot_steps/' + image_num + '.jpg')
        cv.imshow('Frame', image)
        cv.waitKey(1)
        return image
    
    def deleteGifs(self):
        for filename in glob.glob('gifs/*.gif'):
            os.remove(filename)
        
    def reset(self, time_step):
        for filename in glob.glob('robot_steps/*.jpg'):
            os.remove(filename)
        self.maze = self.init_maze
        self.robot_location = self.start_pt
        self.robot_orientation = self.init_orientation//90
        # self.traversed = np.array([])
        # Reset previously traversed locations for the next episode
        self.traversed = []
        self.timestep = 0
        self.total_reward = 0
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
        # TODO: Look into penalty reward for traversed locations and advancing to new spot (maybe swap or alter)
        # NOTE: Do I account for maze edges or walls here?
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # Robot reached the goal
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return 10
        # Robot has already visited this spot
        if (robot_x, robot_y) in self.traversed:
            return -0.6
            # return -0.25
        else:
            # Advanced onto a new spot in the maze, but hasn't reached the goal or gone backwards
            return -0.3
    
    def game_over(self):
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # If rewards value is less than the minimum rewards allowed
        # if self.total_reward < self.min_reward:
        #     return True
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

    def produce_video(self, episodeNum: str, folderPath):
        GIF_DIGITS = 2 # same logic as before, prepends 0's to start of gif
        frames = [Image.open(image) for image in glob.glob(f"robot_steps/*.JPG")]
        frame_one = frames[0]
        os.makedirs(folderPath, exist_ok = True)
        gif_name = folderPath+ '/' + (GIF_DIGITS - len(episodeNum)) * "0" + episodeNum + ".gif"
        frame_one.save(gif_name, format="GIF", append_images=frames,
               save_all=True, duration=300, loop=0)

class ActionError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)