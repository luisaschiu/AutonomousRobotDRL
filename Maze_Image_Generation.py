import cv2 as cv2
import random
import numpy as np
import os, sys
from queue import Queue


class Cell():
    """Cell class that defines each walkable Cell on the grid"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.visited = False
        self.walls = [True, True, True, True] # Left, Right, Up, Down


    def haschildren(self, grid: list) -> list:
        """Check if the Cell has any surrounding unvisited Cells that are walkable"""
        a = [(1, 0), (-1,0), (0, 1), (0, -1)]
        children = []
        for x, y in a:
            if self.x+x in [len(grid), -1] or self.y+y in [-1, len(grid)]:
                continue
            
            child = grid[self.y+y][self.x+x]
            if child.visited:
                continue
            children.append(child)
        return children


def removeWalls(current: Cell, choice: Cell):
    """Removeing the wall between two Cells"""
    if choice.x > current.x:     
        current.walls[1] = False
        choice.walls[0] = False
    elif choice.x < current.x:
        current.walls[0] = False
        choice.walls[1] = False
    elif choice.y > current.y:
        current.walls[3] = False
        choice.walls[2] = False
    elif choice.y < current.y:
        current.walls[2] = False
        choice.walls[3] = False


def drawWalls(grid: list, binGrid: list) -> list:
    """Draw existing walls around Cells"""
    for yi, y in enumerate(grid):
        for xi, x in enumerate(y):
            for i, w in enumerate(x.walls):
                if i == 0 and w:
                    binGrid[yi*2+1][xi*2] = (20,20,20)
                if i == 1 and w:
                    binGrid[yi*2+1][xi*2+2] = (20,20,20)
                if i == 2 and w:
                    binGrid[yi*2][xi*2+1] = (20,20,20)
                if i == 3 and w:
                    binGrid[yi*2+2][xi*2+1] = (20,20,20)
    return binGrid


def drawBorder(grid: list) -> list:
    """Draw a border around the maze"""
    for i, x in enumerate(grid): # Left and Right border
        x[0] = x[len(grid)-1] = (20,20,20)
        grid[i] = x
        
    grid[0] = grid[len(grid)-1] = [(20,20,20) for x in range(len(grid))] # Top and Bottom border
    return grid


def prepareGrid(grid: list) -> list:
    """Turn the grid into RGB values to then be turned into an image"""
    binGrid = []
    for x in range(len(grid)+len(grid)+1):
        if x % 2 == 0:
            binGrid.append([(210, 210 ,210) if x % 2 != 0 else (20,20,20) for x in range(len(grid)+len(grid)+1)])
        else:
            binGrid.append([(210, 210 ,210) for x in range(len(grid)+len(grid)+1)])
    
    binGrid = drawWalls(grid, binGrid)
            
    binGrid = drawBorder(binGrid)

    return binGrid


def prepareImage(grid: list) -> np.ndarray:
    """Turn the grid into a numpy array to then be resized"""
    grid = np.uint8(np.array([np.array(xi) for xi in grid]))

    scale_percent = 1000
    width = int(grid.shape[1] * scale_percent / 100)
    height = int(grid.shape[0] * scale_percent / 100)

    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_AREA)


def generateMaze(size) -> list:
    """Generate a maze of Cell classes to then be turned into an image later"""
    grid = [[Cell(x, y) for x in range(size)] for y in range(size)]
    current = grid[0][0]
    stack = []

    while True:
        current.visited = True
        children = current.haschildren(grid)

        if children:
            choice = random.choice(children)
            choice.visited = True

            stack.append(current)

            removeWalls(current, choice)

            current = choice
        
        elif stack:
            current = stack.pop()
        else:
            return grid


def createImage(grid: list):
    """
    Save the image in the same directory as the Python script under a given name.
    :param grid: The image grid (you'll need to implement the prepareImage function)
    """
    try:
        image = prepareImage(grid)  # Assuming you have implemented prepareImage

        name = input('\nEnter a name to save the image under:\nIt will be stored in the same directory as this Python script.\n>>> ')
        output_path = os.path.join(os.path.dirname(sys.argv[0]), f'{name}.png')
        
        result = cv2.imwrite(output_path, image)
        if result:
            print(f'Status: Image successfully created at {output_path}')
        else:
            print('Status: Something went wrong while saving the image.')
    except Exception as e:
        print(f'Error: {e}')


def rgb_to_bw(binGrid):
    """Turn the RGB grid into a binary grid"""
    bwGrid = np.ones((len(binGrid), len(binGrid[0])), dtype=int)
    for i in range(len(binGrid)):
        for j in range(len(binGrid[i])):
            if binGrid[i][j] == (210, 210, 210):
                bwGrid[i][j] = 0
    return bwGrid

def remove_border(array):
    """Remove the border of a 2D numpy array"""
    return array[1:-1, 1:-1]

def longest_path(maze):
    """Find the longest unbroken path in a numpy maze"""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    max_length = [0]
    start = [0, 0]
    end = [0, 0]

    def dfs(x, y, visited, length):
        if length > max_length[0]:
            max_length[0] = length
            end[0], end[1] = x, y
        visited.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0 and (nx, ny) not in visited:
                dfs(nx, ny, visited, length + 1)
        visited.remove((x, y))

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 0:
                start[0], start[1] = i, j
                dfs(i, j, set(), 0)

    # BFS to find the shortest path from start to end
    visited = np.full(maze.shape, False)
    dist = np.full(maze.shape, np.inf)
    q = Queue()
    q.put(start)
    visited[start[0]][start[1]] = True
    dist[start[0]][start[1]] = 0

    while not q.empty():
        x, y = q.get()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0 and not visited[nx][ny]:
                visited[nx][ny] = True
                dist[nx][ny] = dist[x][y] + 1
                q.put((nx, ny))

    min_steps = dist[end[0]][end[1]]
    return start, end, min_steps if min_steps != np.inf else None
