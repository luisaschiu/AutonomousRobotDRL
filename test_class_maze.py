import numpy as np
from class_maze import Maze


maze_array = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]])

maze = Maze(maze_array, (0,0), (3,3))

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
Maze.show(maze) # Test resetting the maze
Maze.reset(maze)
print(maze.traversed)
Maze.show(maze)