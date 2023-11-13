import class_maze

def show(maze: class_maze):
    plt.grid(True)
    nrows, ncols = self.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
#        maze[self.start_pt[0], self.start_pt[1]] = 0.6
#        maze[self.end_pt[0], self.end_pt[1]] = 0.3
    # maze[2,1] = 0.6
    # canvas = np.copy(maze)
#    for row,col in maze.visited:
#        canvas[row,col] = 0.6
#    rat_row, rat_col, _ = maze.state
#    canvas[rat_row, rat_col] = 0.3   # rat cell
#    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    # img = plt.imshow(canvas, interpolation='none', cmap='gray')

    img = plt.imshow(self.maze, interpolation='none', cmap='binary')
    img = plt.show()
    return img
