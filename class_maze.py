import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import glob
import random
from PIL import Image

class Maze:
    def __init__(self, maze:np.array, marker_filepath:str, start_pt: tuple, goal_pt: tuple, start_orientation:int, slip_flag = False):
        self.init_maze = np.copy(maze)
        self.maze = maze
        self.robot_location = start_pt
        self.init_orientation = start_orientation
        self.robot_orientation = start_orientation//90
        self.marker = mpimg.imread(marker_filepath)
        # NOTE: This is assuming the maze is a square:
        # print(self.maze.shape)
        # print(start_pt)
        if start_pt[0] < 0.0 or start_pt[0] > (self.maze.shape[0]-1) or start_pt[1] < 0.0 or start_pt[1] > (self.maze.shape[0]-1):
            raise MazeError("Defined start point is out of boundaries. Ensure you choose a valid start point within the maze boundaries.")
        elif self.maze[start_pt[1], start_pt[0]] == 1.0:
            raise MazeError("Defined start point cannot be a maze wall. Ensure you choose a free space within the maze.")
        self.start_pt = start_pt
        if goal_pt[0] < 0.0 or goal_pt[0] > (self.maze.shape[0]-1) or goal_pt[1] < 0.0 or goal_pt[1] > (self.maze.shape[0]-1):
            raise MazeError("Defined goal point is out of boundaries. Ensure you choose a valid goal point within the maze boundaries.")
        elif self.maze[goal_pt[1], goal_pt[0]] == 1.0:
            raise MazeError("Defined goal point cannot be a maze wall. Ensure you choose a free space within the maze.")
        self.goal_pt = goal_pt
        self.traversed = []
        self.total_reward = 0
        self.slip_flag = slip_flag

    def show(self):
        # print(self.robot_location)
        # print(self.start_pt)
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
        self.maze[self.start_pt[1], self.start_pt[0]] = 0.3
        self.maze[self.goal_pt[1], self.goal_pt[0]] = 0.6
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


    def show_game(self, q_values):
        # print(self.robot_location)
        # print(self.start_pt)
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
        self.maze[self.start_pt[1], self.start_pt[0]] = 0.3
        self.maze[self.goal_pt[1], self.goal_pt[0]] = 0.6
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
        # if q_values[0][1] != float('-inf'):
        #     self.maze[self.robot_location[1]+1, self.robot_location[0]] = q_values[0][1]
        img = plt.imshow(self.maze, interpolation='none', cmap='binary', alpha = 0.8)
        # plt.title('Maze Walls')


        # Plot Q-values using hot colormap
        # maze_copy = self.maze.copy()
        # print(self.maze.shape)
        # ax2 = plt.gca()
        # ax2.set_xticks(np.arange(0.5, nrows, 1))
        # ax2.set_yticks(np.arange(0.5, ncols, 1))
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        test_numpy = np.ones((nrows, ncols))
        test_numpy[1, 0]=0.8
        # test_numpy[1, 1]=0.2
        # if q_values[0][1] != float('-inf'):
        #     test_numpy[self.robot_location[1]+1, self.robot_location[0]] = q_values[0][1]
        print(test_numpy)
        plt.imshow(test_numpy, cmap='hot', alpha = 0.2)  # Set alpha for transparency
        # plt.title('Q-Values')
        #TODO: Create a 4x4 numpy array with the q-values corresponding to the current robot position, 
        # with everything else equal to 1 for the q-value heat map.
        plt.colorbar()  # Add color bar for Q-values
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
        self.maze[self.start_pt[1], self.start_pt[0]] = 0.3
        self.maze[self.goal_pt[1], self.goal_pt[0]] = 0.6
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
        if self.slip_flag:
            choices = ['slip', 'no slip']
            probabilities = [0.2, 0.8]  # Adjust these probabilities as needed
            # Generate a random choice based on the defined probabilities
            random_choice = random.choices(choices, weights=probabilities, k=1)[0]
            # print('random choice: ', random_choice)
            if random_choice == 'slip':
                # print('slip')
                return 'slip'
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
                return 'no slip'
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
                return 'no slip'
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
                return 'no slip'
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
                return 'no slip'


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
    
    def manhattan_distance(self, start_x, start_y, end_x, end_y):
        # Calculate Manhattan distance between two grid cells
        return abs(start_x - end_x) + abs(start_y - end_y)
    
    def get_reward_heuristics(self):
        # TODO: Look into penalty reward for traversed locations and advancing to new spot (maybe swap or alter)
        # NOTE: Do I account for maze edges or walls here?
        robot_x, robot_y = self.robot_location[0], self.robot_location[1]
        # Robot reached the goal
        if robot_x == self.goal_pt[0] and robot_y == self.goal_pt[1]:
            return 10
        # Robot has already visited this spot
        # if (robot_x, robot_y) in self.traversed:
        # # if len(self.traversed) != 0 and (robot_x, robot_y) == self.traversed[-1]:
        # # if (robot_x, robot_y) == self.traversed[-1]:
        #     return -0.8
            # return -0.25
        else:
            # Advanced onto a new spot in the maze, but hasn't reached the goal or gone backwards
            heuristic = self.manhattan_distance(robot_x, robot_y, self.goal_pt[0], self.goal_pt[1])
            norm_heuristic = heuristic/self.manhattan_distance(self.start_pt[0], self.start_pt[1], self.goal_pt[0], self.goal_pt[1])
            # return -0.4*norm_heuristic
            return -(heuristic**2)

    
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
        slip_condition = self.move_robot(action)
        if slip_condition == 'slip':
            reward = 0
            # print('reward: ', reward)
        else:
            reward = self.get_reward()
            # print('reward: ', reward)
        self.total_reward += reward
        game_over = self.game_over()
        # self.time_step += 1
        new_state_img = self.generate_img(time_step)
        return (new_state_img, reward, game_over)
    
    def take_action_heuristics(self, action: str, time_step):
        slip_condition = self.move_robot(action)
        if slip_condition == 'slip':
            reward = 0
            # print('reward: ', reward)
        else:
            reward = self.get_reward_heuristics()
            # print('reward: ', reward)
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

def deleteGifs(folder_type: str):
    if folder_type == "gameplay":
        for filename in glob.glob('gameplay_episode_videos/*.gif'):
            os.remove(filename)
    elif folder_type == "training":
        for filename in glob.glob('training_episode_videos/*.gif'):
            os.remove(filename)

class MazeError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)

class ActionError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)