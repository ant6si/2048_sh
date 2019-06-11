import os,time, random, math
# import pygame
os.environ['SDL_AUDIODRIVER'] = 'dsp'

from copy import deepcopy
# from pprint import pprint
# import numpy as np
# import _2048
# from _2048.game import Game2048
# from _2048.manager import GameManager
from FeatureHandler import *
from multiprocessing import Process, Queue, Pool
from operation import *

# define events
# EVENTS = [
#   pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}),   # UP
#   pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}),  # RIGHT
#   pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}),  # DOWN
#   pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT})  # LEFT
# ]

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

PROCESS_NUM = 10
BATCH_SIZE = PROCESS_NUM
LEARNING_RATE =  0.005  # 0.01
DEPTH = 1
weight_file_name = "../results/comb1.pickle"
result_file_name = "../results/comb1.txt"

episodes = Queue()
# f_queue = Queue()
tokens = Queue()


def print_grid(grid):
    """
    print board
    """
    for r in range(4):
        for c in range(4):
            print("{}".format(grid[r][c]), end=" ")
        print()
    print()

def get_initial_grid():
    """
    initialize the board
    """
    grid = [ [ 0 for i in range(4)] for j in range(4) ]
    add_random_tile(grid)
    add_random_tile(grid)
    return grid

def free_cells(grid):
    """
    count number of free cells in the grid
    """
    return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]

def add_random_tile(grid):
    """
    randomly add 2 or 4 value to the board
    """
    free_cell_list = free_cells(grid)
    assert len(free_cell_list) !=0

    r1 = random.randrange(0,len(free_cell_list))
    rand_loc = free_cell_list[r1]

    assert grid[rand_loc[1]][rand_loc[0]] == 0
    r2 = random.random()
    if r2 < 0.1:
        grid[rand_loc[1]][rand_loc[0]] = 4
    else:
        grid[rand_loc[1]][rand_loc[0]] = 2


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
            sum: acquired score by the movement
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


def evaluation(grid, f_handler):
    """
    evaluate the score of the grid
    f_handler: feature set
    """
    grid = np.array(grid)
    grid_score = f_handler.getValue(grid)
    return grid_score


def findBestMove(grid, f_handler, depth=0):
    """
    Find Best Move
    Expectimax Search for depth 1
    """
    best_score = -np.inf
    best_action = None
    best_moved_grid = deepcopy(grid)
    for action in range(4):
        given_grid = deepcopy(grid)
        moved_grid, moved, _ = move(given_grid, action)#action=action)

        if not moved:
            continue

        new_score = expected_random_tile_score(moved_grid, f_handler, depth+1)
        if new_score >= best_score:
            best_moved_grid = deepcopy(moved_grid)
            best_score = new_score
            best_action = action
    # if there is no way to move, return origin moved_board
    return best_action, best_score, best_moved_grid # moved_grid: mboard


def expected_random_tile_score(grid,f_handler, depth=0):
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
        return evaluation(grid, f_handler)

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

def play_game_forever():
    """
    play game until the game is over
    """
    while True:
        tokens.get()
        # print("token!")
        my_f_handler = FeatureHandler()

        my_f_handler.loadWeights(weight_file_name)
        # print("Load weight! Successful!")

        # line_weight_0 = my_f_handler.featureSet[0].getWeight()[0][0]
        # print("PLAYER_LOADED: Length of line weight 0: {}\n".format(len(line_weight_0)))
        play_game(my_f_handler)


def play_game(f_handler):
    """
    play game until the game is over, and save the board states, reward, and score as an episode
    f_handler: feature set
    """
    # print("data_path: {}".format(data_dir))
    board_chain = []
    reward_chain = []
    grid = get_initial_grid()
    count = 0
    game_score = 0
    canMove = True
    while canMove:
        # t1 = time.time()
        count+=1
        #print("count: {}".format(count))
        old_grid = deepcopy(grid)
        best_action, best_score, moved_grid = findBestMove(old_grid, f_handler)
        board_chain.append(moved_grid)
        # t2 = time.time()
        if best_action is None:
            reward = 0
            reward_chain.append(reward)
            #print('The end. \n Score:{} / Max num: {}'.format(manager.game.score, np.max(manager.game.grid)))
            canMove = False
            break
        #print(best_action)
        _, _, reward = move(grid, best_action)
        game_score += reward
        reward_chain.append(reward)
        add_random_tile(grid)
        #pprint(manager.game.grid, width=30)
        #print(manager.game.score)
        # t3 = time.time()
        # print("Find best action: {}s , dispatch action: {}s".format(t2-t1, t3-t2))
    episode = [board_chain, reward_chain, game_score]
    episodes.put(episode)
    return game_score

def batch_update_forever(batch_count, f_handler):
    """
    update weight in batch size
    batch_count: batch size
    f_handler: feature set
    """
    many_batch_sum = 0
    many_batch_max = 0
    many_batch_avg = 0
    many_batch_size = 10
    while True:
        batch_count+=1
        batch_start = time.time()
        [tokens.put('token') for i in range(PROCESS_NUM)]
        batch_avg, batch_max = batch_update(batch_count, f_handler)

        """Print batch results on console"""
        print("Batch: {} / Time: {} / Avg: {} / Max {}".format(batch_count, time.time() - batch_start
                                                               , batch_avg, batch_max))

        """Print ten-batch results on """
        many_batch_sum += batch_avg * BATCH_SIZE
        if many_batch_max < batch_max:
            many_batch_max = batch_max

        if batch_count % many_batch_size == 0:
            # print and reset
            many_batch_avg = many_batch_sum / float(many_batch_size * BATCH_SIZE)
            with open(result_file_name, "a") as f:
                f.write("Batch count\t\t{}\t\tAVG\t\t{}\t\tMAX\t\t{}\n".format(batch_count, many_batch_avg
                                                                               , many_batch_max))
            many_batch_max, many_batch_avg, many_batch_sum = 0, 0, 0

def batch_update(batch_count, f_handler):
    """
    update weight in batch size
    batch_count: batch size
    f_handler: feature set
    """
    batch_size=BATCH_SIZE
    dict = {}
    batch_score_sum=0
    batch_max_score = 0
    for i in range(batch_size):
        episode = episodes.get()
        # print("Get {}-th episode, Game score: {}".format(i+1, episode[2]))
        batch_score_sum += episode[2]
        if batch_max_score < episode[2]:
            batch_max_score = episode[2]
        updateEvaluation(dict, episode, f_handler)
    batch_score_avg = batch_score_sum / float(batch_size)
    # print("Get {} episodes, start update, Average score: {}".format(batch_size, batch_score_avg))

    # print("Size of dict: {}".format(len(dict)))
    for tuple_moved_board in dict.keys():
        moved_board = tuple_to_list(tuple_moved_board)
        delta = dict.get(tuple_moved_board)[0]
        f_handler.updateValue(np.array(moved_board), delta)
    # print("---update done, save the latest weights---")

    """DEBUG"""
    # line_weight_0 = f_handler.featureSet[0].getWeight()[0]
    # print("update: Length of line weight 0: {}\n".format(len(line_weight_0)))

    f_handler.saveWeights(weight_file_name)
    # [f_queue.put(f_handler) for iter in range(PROCESS_NUM)]

    # if max_avg <= batch_score_avg:
    #     print("---Find Maximum Weights and Save it---")
    #     print("Before max: {}/ After max: {}".format(max_avg, batch_score_avg))
    #     max_avg = batch_score_avg
    #     # weight_indices = np.where(f_handler.featureSet[0].getWeight() != 0)
    #     f_handler.saveWeights("saved_best_weights.pickle")
    del dict
    # print("update done")
    return batch_score_avg, batch_max_score

    # return max_avg

def updateEvaluation(dict, episode, f_handler):
    """
    read in whole episodes and update the weight by using TD learning
    episode: sequence board state
    f_handler: feature set
    """
    board_chain = episode[0]
    reward_chain = episode[1]
    chain_len = len(board_chain)
    lamb=0.5
    for i in range(chain_len):
        size = 5
        G_t_lambda = 0
        j = 0
        sum_of_weight=0
        while j < size and (i+j) < chain_len:
            if (j != size-1) or (i+j != chain_len-1):
                weight = (1-lamb) * pow(lamb, j*1.0)
                sum_of_weight += weight
            else:
                # weight = 1 - sum_of_weight
                weight = pow(lamb, j*1.0)
            reward = reward_chain[i+j]

            #reward = v[i+j][1]
            G_t_lambda += weight * (reward + evaluation(board_chain[i + j], f_handler))
            # score += weight * (reward + evaluation(v[i+j][2]))
            j += 1
        # print("score: {}, value: {}".format(G_t_lambda, evaluation(board_chain[i])))
        delta = LEARNING_RATE * (G_t_lambda - evaluation(board_chain[i], f_handler))
        # print(delta);
        tuple_board = list_to_tuple(board_chain[i])
        if tuple_board in dict:
            # value = [ average delta, count]
            value = dict.get(tuple_board)
            value[0] = float(value[0]) * value[1] / float(value[1]+1) + 1.0 / (value[1]+1) * delta
            value[1] += 1
        else:
            dict[tuple_board] = [delta, 1]
        # f_handler.updateValue(np.array(board_chain[i]), delta)
        # f_handler.updateValue(v[i][2], LEARNING_RATE * (score - f_handler.getValue(v[i][2])))

def isGameDone(players):
    """
    check if the game is over
    """
    is_all_dead = True
    for index in range(len(players)):
        player = players[index]
        if player.is_alive():
            print("{}-th player is not done.".format(index))
            is_all_dead = False
    return is_all_dead

def proc_func(arg):
    """
    mapping functions
    """
    arg[0](*arg[1])

def remove_state_files():
    """
    detele state files
    """
    for file_num in range(PROCESS_NUM):
        rm_target = './save/2048.{}.state'.format(file_num)
        if os.path.isfile(rm_target):
            os.remove(rm_target)
            print("remove {}".format(rm_target))

if __name__ == "__main__":
    f_handler = FeatureHandler()
    # """Load saved-weight"""
    # print("before load weight: {}".format(f_handler.featureSet[0].getWeight()))
    # print("---Load saved weights---")
    # f_handler.loadWeights(weight_file_name)
    # line_weight = f_handler.featureSet[0].getWeight()
    # # print("after load weight: {}".format(load_weight_indices))
    # with open('weight_check.txt', 'w') as f:
    #     for i in range(8):
    #         a = int(i/4)
    #         b = int(i%4)
    #         f.write("{}\n".format(line_weight[a][b]))

    '''TD Learning Using Multiprocess W Pool'''
    batch_count = 0
    # [f_queue.put(f_handler) for iter in range(PROCESS_NUM)]
    # remove_state_files()
    pool = Pool(processes=PROCESS_NUM + 1)
    funcs = PROCESS_NUM * [(play_game_forever, ( ))] \
            + [(batch_update_forever, (batch_count, f_handler))]
    pool.map(proc_func, funcs)

