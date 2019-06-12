import os, pygame, time, random, math
os.environ['SDL_AUDIODRIVER'] = 'dsp'
import sys
sys.path.append('/src/')
from copy import deepcopy
from pprint import pprint
import numpy as np
import _2048
from _2048.game import Game2048
from _2048.manager import GameManager
from FeatureHandler import *
from operation import *
import queue
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--line",  default=False, type=bool)
parser.add_argument("--rec",  default=False, type=bool)
parser.add_argument("--axe",  default=False, type=bool)
parser.add_argument("--max",  default=False, type=bool)
parser.add_argument("--merge",  default=False, type=bool)
parser.add_argument("--layer",  default=False, type=bool)
parser.add_argument("--distinct",  default=False, type=bool)
parser.add_argument("--empty",  default=False, type=bool)
parser.add_argument("--num", required=True, type=int)
parser.add_argument("--lr",  default=0.01, type=float)

args = parser.parse_args()

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

PROCESS_NUM = 1
BATCH_SIZE = PROCESS_NUM
LEARNING_RATE = args.lr  # 0.01
DEPTH = 1

f_handler = FeatureHandler(args)
episodes = queue.Queue()
f_queue = queue.Queue()
# tokens = Queue()

weights_filename = "../results/one_saved_latest_weights_" + str(args.num) + ".pickle"
result_filename = "../results/one_result_" + str(args.num) + ".txt"

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


def evaluation(grid):
    """
    evaluate the board
    """
    # evaluate the score of the grid
    grid = np.array(grid)
    grid_score = f_handler.getValue(grid)
    return grid_score


def findBestMove(grid, depth=0):
    """
    compute the value of next states and choose the best action which will get the best state value
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

        new_score = expected_random_tile_score(moved_grid, depth+1)
        if new_score >= best_score:
            best_moved_grid = deepcopy(moved_grid)
            best_score = new_score
            best_action = action
    # if there is no way to move, return origin moved_board
    return best_action, best_score, best_moved_grid # moved_grid: mboard


def expected_random_tile_score(grid, depth=0):
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


def play_game():
    """
    play game until the game is over, and save the board state as episode
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
        best_action, best_score, moved_grid = findBestMove(old_grid)
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

def batch_update():
    """
    update weight in batch size
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
        updateEvaluation(dict, episode)
    batch_score_avg = batch_score_sum / float(batch_size)
    # print("Get {} episodes, start update, Average score: {}".format(batch_size, batch_score_avg))

    # print("Size of dict: {}".format(len(dict)))
    for tuple_moved_board in dict.keys():
        moved_board = tuple_to_list(tuple_moved_board)
        delta = dict.get(tuple_moved_board)[0]
        f_handler.updateValue(np.array(moved_board), delta)
    # print("---update done, save the latest weights---")

    """DEBUG"""
    # line_weight_0 = f_handler.featureSet[0].getWeight()[0][0]
    # print("update: Length of line weight 0: {}\n".format(len(line_weight_0)))

    f_handler.saveWeights(weights_filename)

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

def updateEvaluation(dict, episode):
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
            G_t_lambda += weight * (reward + evaluation(board_chain[i + j]))
            # score += weight * (reward + evaluation(v[i+j][2]))
            j += 1
        # print("score: {}, value: {}".format(G_t_lambda, evaluation(board_chain[i])))
        delta = LEARNING_RATE * (G_t_lambda - evaluation(board_chain[i]))
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

    # """Load saved-weight"""
    # print("before load weight: {}".format(f_handler.featureSet[0].getWeight()))
    # print("---Load saved weights---")
    # f_handler.loadWeights("one_saved_latest_weights.pickle")
    # line_weight = f_handler.featureSet[0].getWeight()
    # # print("after load weight: {}".format(load_weight_indices))
    # with open('weight_check.txt', 'w') as f:
    #     for i in range(8):
    #         a = int(i/4)
    #         b = int(i%4)
    #         f.write("{}\n".format(line_weight[a][b]))

    '''TD Learning Without Using Multiprocess'''
    count = 0
    score_sum = 0
    tot_max = 0
    partial_max = 0
    tw_max = 0
    tw_sum = 0
    training_st = time.time()
    while True:
        count += 1
        st = time.time()
        score = play_game()
        st_p = time.time()
        # statistics
        score_sum += score
        tw_sum += score
        if tot_max < score:
            tot_max = score
        if tw_max < score:
            tw_max = score
        if partial_max < score:
            partial_max = score


        batch_update()
        et = time.time()

        if count % 20 == 0:
            tw_avg = tw_sum / 20.0
            tw_sum = 0
            time_elapse = time.time()-training_st
            print("time\t{:.2f}\tcount\t{}\tAVG(20)\t{}\tMAX(20)\t{} MAX(tot)\t{}".format(time_elapse,count,tw_avg, tw_max, tot_max))
            tw_max = 0


        if count %100 == 0:
            avg_score = score_sum / 100.0
            score_sum = 0
            partial_max = 0
            time_elapse = time.time() - training_st
            with open(result_filename, "a") as f:
                f.write("time\t\t{:.2f}\t\tcount\t\t{}\t\tAVG\t\t{}\t\tMAX\t\t{}\n".format(time_elapse, count, avg_score
                                                                               , partial_max))

