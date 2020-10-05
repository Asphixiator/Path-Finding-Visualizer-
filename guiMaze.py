# Adding and testing bfs algo.

import pygame as pg
import sys
from collections import deque
from queue import PriorityQueue

pg.init()
WIDTH = 700
BLACK = (11, 11, 11)
GREEN = (90, 200, 80)
WHITE = (240, 240, 240)
RED = (240, 50, 50)
ORANGE = (255, 165, 0)
BLUE = (172, 240, 120)
PURPLE = (128, 0, 128)
GREY = (211, 211, 211)

WIN = pg.display.set_mode((WIDTH, WIDTH))
pg.display.set_caption("Pygame Mazy")


class Node:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_close(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == BLUE

    def reset(self):
        self.color = WHITE

    def make_close(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = BLUE

    def make_start(self):
        self.color = ORANGE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pg.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbours = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbours.append(grid[self.row][self.col - 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbours.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col + 1])

    def __lt__(self, other):
        return False


def trace_path(came_from, cur, draw):
    print("Entered trace path")
    while cur in came_from:
        cur = came_from[cur]
        cur.make_path()
        draw()
    print("Exiting trace path")


def dfs(draw, grid, start, end):
    frontier = list()
    came_from = dict()

    frontier.append(start)
    while len(frontier) > 0:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        cur = frontier.pop()
        try:
            if cur == end:
                trace_path(came_from, cur, draw)
                end.make_end()
                # print("End found!!")
                return True
        except:
            return True

        for neighbour in cur.neighbours:
            if not neighbour.is_close():
                came_from[neighbour] = cur
                neighbour.make_open()
                frontier.append(neighbour)

        draw()

        if cur != start:
            cur.make_close()

    return False


def bfs(draw, grid, start, end):
    frontier = deque()
    came_from = dict()

    frontier.append(start)
    while frontier:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        cur = frontier.popleft()
        if cur == end:
            trace_path(came_from, cur, draw)
            end.make_end()
            return True

        for neighbour in cur.neighbours:
            if not neighbour.is_close():
                came_from[neighbour] = cur
                neighbour.make_open()
                frontier.append(neighbour)

        draw()

        if cur != start:
            cur.make_close()

    return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return abs(x2 - x1) + abs(y2 - y1)


def astar(draw, grid, start, end):
    count = 0
    frontier = PriorityQueue()
    frontier.put((0, count, start))
    came_from = dict()
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # To keep track of items in the priority queue. Priority queue has no functionality to check wether
    # an element exists in the queue

    frontier_hash = {start}

    while not frontier.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        cur = frontier.get()[2]
        frontier_hash.remove(cur)

        if cur == end:
            trace_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbour in cur.neighbours:
            temp_g_score = g_score[cur] + 1

            if temp_g_score < g_score[neighbour]:
                g_score[neighbour] = temp_g_score
                came_from[neighbour] = cur
                f_score[neighbour] = temp_g_score + \
                    h(neighbour.get_pos(), end.get_pos())

                if neighbour not in frontier_hash:
                    count += 1
                    frontier.put((f_score[neighbour], count, neighbour))
                    frontier_hash.add(neighbour)
                    neighbour.make_open()

        draw()

        if cur != start:
            cur.make_close()

    return False


def make_grid(rows, width):
    grid = []

    # finding width for each of the cubes
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows

    for i in range(rows):
        pg.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pg.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win, rows, width)
    pg.display.update()


def mouse_position(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 50

    grid = make_grid(ROWS, width)
    start = None
    end = None

    run = True

    while run:
        draw(win, grid, ROWS, width)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            # checking if mouse button is pressed(Left mouse button)
            if pg.mouse.get_pressed()[0]:
                pos = pg.mouse.get_pos()
                row, col = mouse_position(pos, ROWS, width)
                node = grid[row][col]

                if not start and node != end:
                    start = node
                    start.make_start()

                elif not end and node != start:
                    end = node
                    end.make_end()

                elif node != start and node != end:
                    node.make_barrier()

            # Right mouse button
            elif pg.mouse.get_pressed()[2]:
                pos = pg.mouse.get_pos()
                row, col = mouse_position(pos, ROWS, width)
                node = grid[row][col]
                node.reset()

                if node == start:
                    start = None

                elif node == end:
                    end = None

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)

                    # astar(lambda: draw(win, grid, ROWS, width),
                    #       grid, start, end)

                    # bfs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                    dfs(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    # print("Returned back safely")

                if event.key == pg.K_ESCAPE:
                    start = None
                    end = None
                    grid = make_grid(ROWS, WIDTH)

    pg.quit()


main(WIN, WIDTH)


# class MazeSolver:
# 	def __init__(self):
# 		self.frontier = deque()
# 		self.visited = list()
# 		self.solution = dict()
# 		self.start = []
# 		self.end = []
# 		self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]	# Down, Right, Up, Left

# 	def findStart(self):
# 		for i, row in enumerate(maze):
# 			for j, col in enumerate(list(row)):
# 				if col == "X":
# 					self.start = [i, j]

# 	def findEnd(self):
# 		for i, row in enumerate(maze):
# 			for j, col in enumerate(list(row)):
# 				if col == "O":
# 					self.end = [i, j]

# 	def exploreNeighbours(self, curpos):
# 		for i in self.directions:
# 			nextpos = list(map(sum, zip(curpos, i)))
# 			if (nextpos[0] >= len(maze)) or (nextpos[1] >= len(list(maze[0]))):
# 				continue
# 			if (nextpos[0] < 0) or (nextpos[1] < 0):
# 				continue
# 			if nextpos in self.visited:
# 				continue
# 			if maze[nextpos[0]][nextpos[1]] == '#':
# 				continue

# 			self.visited.append(curpos)
# 			self.solution[nextpos[0], nextpos[1]] = curpos[0], curpos[1]
# 			self.frontier.append(nextpos)

# 	def solve(self):
# 		self.findStart()
# 		self.findEnd()
# 		self.frontier.append(self.start)
# 		self.solution[self.start[0], self.start[1]] = self.start[0], self.start[1]

# 		while len(self.frontier) > 0:
# 			curpos = self.frontier.popleft()
# 			self.exploreNeighbours(curpos)
