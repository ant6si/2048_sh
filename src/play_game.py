import os, pygame, time, random, math
from copy import deepcopy
from pprint import pprint
import numpy as np
import _2048
from _2048.game import Game2048
from _2048.manager import GameManager
from FeatureHandler import *

# define events
EVENTS = [
  pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}),   # UP
  pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}),  # RIGHT
  pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}),  # DOWN
  pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT})  # LEFT
]

CELLS = [
  [(r, c) for c in range(4) for r in range(4)],  # UP
  [(r, c) for r in range(4) for c in range(4 - 1, -1, -1)],  # RIGHT
  [(r, c) for c in range(4) for r in range(4 - 1, -1, -1)],  # DOWN
  [(r, c) for r in range(4) for c in range(4)],  # LEFT
]

GET_DELTAS = [
  lambda r, c: ((i, c) for i in range(r + 1, 4)),  # UP
  lambda r, c: ((r, i) for i in range(c - 1, -1, -1)),  # RIGHT
  lambda r, c: ((i, c) for i in range(r - 1, -1, -1)),  # DOWN
  lambda r, c: ((r, i) for i in range(c + 1, 4))  # LEFT
]

board_chain = []
reward_chain = []
f_handler = FeatureHandler()
LEARNING_RATE = 0.025 #0.00025
DEPTH=3;


def free_cells(grid):
    """
    find the free cells and returns the location of free tiles
    :param grid:
    :return: list of locations of the free cells
    """
    # count number of free cells in the grid
    return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]


def move(grid, action):
    """
        check if the grid can move follow the action.
        If it can move it will return the reward and moved grid
        args
            grid: current state of the grid
            action: the way we want to move
        return
            grid: moved grid after action (random tile not added)
            moved: number of movement (but, use it like boolean)
            sum: new score by the movement
    """
    moved, sum = 0, 0
    for row, column in CELLS[action]:
        for dr, dc in GET_DELTAS[action](row, column):
            # If the current tile is blank, but the candidate has value:
            if not grid[row][column] and grid[dr][dc]:
                # Move the candidate to the current tile.
                grid[row][column], grid[dr][dc] = grid[dr][dc], 0
                moved += 1
            if grid[dr][dc]:
                # If the candidate can merge with the current tile:
                if grid[row][column] == grid[dr][dc]:
                    grid[row][column] *= 2
                    grid[dr][dc] = 0
                    sum += grid[row][column]
                    moved += 1
                # When hitting a tile we stop trying.
                break
    return grid, moved, sum


def evaluation(grid):
    """
    compute the score of the grid using feature handler
    :param grid:
    :return:grid_score
    """
    # evaluate the score of the grid
    grid = np.array(grid)
    grid_score = f_handler.getValue(grid)
    return grid_score


def findBestMove(grid, depth=0):
    """
    Find the best move with expectimax of selected deoth
    :param grid:
    :param depth:
    :return:best action
    :return best_score
    :return best_moved_grid
    """
    # Find Best Move
    # Expectimax Search for depth 1
    best_score = -np.inf
    best_action = None
    best_moved_grid = deepcopy(grid)
    for action in range(4):
        given_grid = deepcopy(grid)
        moved_grid, moved, _ = move(given_grid, action)#action=action)

        if not moved:
            continue

        new_score = add_new_tiles(moved_grid, depth+1)
        if new_score >= best_score:
            best_moved_grid = deepcopy(moved_grid)
            best_score = new_score
            best_action = action
    # if there is no way to move, return origin moved_board
    return best_action, best_score, best_moved_grid # moved_grid: mboard


def add_new_tiles(grid, depth=0):
    """
    compute expectimax score of depth<=3
    arg
        grid: given grid (moved grid)
        depth: depth of expectation (for expectimax)
    return
        score: expected score of the given grid when it gets random tile.
    """
    fcs = free_cells(grid)
    n_empty = len(fcs)
    if depth >= DEPTH:
        return evaluation(grid)

    sum_score = 0

    for x, y in fcs:
        for v in [2, 4]:
            new_grid = deepcopy(grid)
            new_grid[y][x] = v

            _, new_score, _ = findBestMove(new_grid, depth+1)

            if v == 2:
                new_score *= (0.9 / n_empty)
            else:
                new_score *= (0.1 / n_empty)
            sum_score += new_score

    return sum_score


def play_game(game_class=Game2048, title='2048!', data_dir='save'):
    """
    play one game using with game agent
    :param game_class:
    :param title:
    :param data_dir:
    :return: final score
    :return max_tile
    """
    # print("data_path: {}".format(data_dir))
    final_score = 0
    max_tile = -1
    pygame.init()
    pygame.display.set_caption(title)
    pygame.display.set_icon(game_class.icon(32))
    clock = pygame.time.Clock()
    os.makedirs(data_dir, exist_ok=True)
    screen = pygame.display.set_mode((game_class.WIDTH, game_class.HEIGHT))
    # screen = pygame.display.set_mode((50, 20))
    manager = GameManager(Game2048, screen,
                  os.path.join(data_dir, '2048.score'),
                  os.path.join(data_dir, '2048.%d.state'))
    # faster animation
    manager.game.ANIMATION_FRAMES = 1
    manager.game.WIN_TILE = 999999

    # game loop
    tick = 0
    running = True

    count = 0
    while running:
        clock.tick(120)
        tick += 1

        if tick % 2 == 0:
            count+=1
            #print("count: {}".format(count))
            old_grid = deepcopy(manager.game.grid)
            best_action, best_score, moved_grid = findBestMove(old_grid)

            if best_action is None:
                max_tile = np.max(old_grid)
                final_score = manager.game.score
                #print('The end. \n Score:{} / Max num: {}'.format(manager.game.score, np.max(manager.game.grid)))
                break

            #print(best_action)
            e = EVENTS[best_action]
            manager.dispatch(e)
            #pprint(manager.game.grid, width=30)
            #print(manager.game.score)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                manager.dispatch(event)

        manager.draw()

    manager.close()
    return final_score, max_tile



if __name__ == "__main__":
    weight_file = "../pickles/comb3_0.005.pickle"
    result_file = "../result/random_result.txt"
    print("Load saved weights: {}".format(weight_file))
    f_handler.loadWeights(weight_file)

    count = 0
    max_score = 0
    while count<100:
        count += 1
        score, max_tile = play_game()
        print("iter: {} / score: {}".format(count, score))
        with open(result_file, "a") as f:
            f.write(" {}\t\t{}\t\t{}\n".format(count, score, max_tile))
            print("written!")
        #print("final score: {} ".format(score))
    pygame.quit()
    print("End Successfully")



