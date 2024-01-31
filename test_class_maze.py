import numpy as np
from class_maze import Maze


maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]])
marker_filepath = "images/marker8.jpg"
maze = Maze(maze_array, marker_filepath, (0,0), (3,3), 2)

Maze.show(maze)
print(maze.traversed)
Maze.move_robot(maze, "DOWN")
print(maze.traversed)
Maze.show(maze)
Maze.move_robot(maze, "DOWN") # Yield wall detected error
print(maze.traversed)
Maze.move_robot(maze, "RIGHT")
print(maze.traversed)
Maze.move_robot(maze, "RIGHT")
Maze.move_robot(maze, "RIGHT")
print(maze.traversed)
Maze.move_robot(maze, "RIGHT") # Yield maze edge detected error
print(maze.traversed)
Maze.show(maze)
# Test resetting the maze
print("Resetting Maze")
Maze.reset(maze)
print(maze.traversed)
Maze.show(maze)