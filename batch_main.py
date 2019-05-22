import os, pygame, time, random, math
os.environ['SDL_AUDIODRIVER'] = 'dsp'
from copy import deepcopy
from pprint import pprint
import numpy as np
import _2048
from _2048.game import Game2048
from _2048.manager import GameManager
from FeatureHandler import *
from multiprocessing import Process, Queue

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

PROCESS_NUM = 3
BATCH_SIZE = 3
f_handler = FeatureHandler()
LEARNING_RATE = 0.025 #0.00025
DEPTH=3;


def free_cells(grid):
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
    # evaluate the score of the grid
    grid = np.array(grid)
    grid_score = f_handler.getValue(grid)
    return grid_score


def findBestMove(grid, depth=0):
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

def play_game_N_times (queue, N=100000):
    for i in range(N):
        play_game(queue)

def play_game(queue, game_class=Game2048, title='2048!', data_dir='save'):
    # print("data_path: {}".format(data_dir))
    board_chain = []
    reward_chain = []
    final_score = 0
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
            old_score = manager.game.score
            best_action, best_score, moved_grid = findBestMove(old_grid)
            board_chain.append(moved_grid)

            if best_action is None:
                final_score = manager.game.score
                new_score = manager.game.score
                reward = new_score - old_score
                reward_chain.append(reward)
                #print('The end. \n Score:{} / Max num: {}'.format(manager.game.score, np.max(manager.game.grid)))
                break

            #print(best_action)
            e = EVENTS[best_action]
            manager.dispatch(e)
            new_score = manager.game.score
            reward = new_score-old_score
            reward_chain.append(reward)
            #pprint(manager.game.grid, width=30)
            #print(manager.game.score)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                manager.dispatch(event)

        manager.draw()
    episode = [board_chain, reward_chain, final_score]
    queue.put(episode)
    manager.close()
    pygame.quit()
    return final_score

def print_episode(queue):
    episode = queue.get()
    board_chain = episode[0]
    reward_chain = episode[1]
    print("board_length: {}, reward_length: {}".format(len(board_chain), len(reward_chain)))
def print_episode_forever(queue):
    while True:
        print_episode(queue)

def list_to_tuple(l):
    # only work for dimension 2
    length = len(l)
    t_list = []
    for i in range(length):
        t_list.append(tuple(l[i]))
    t = tuple(t_list)
    return t

def tuple_to_list(t):
    length = len(t)
    l = []
    for i in range(length):
        l.append(list(t[i]))
    return l


def batch_update_forever(queue):
    batch_count = 0
    max_avg_score = 0
    while True:
        batch_count += 1
        max_avg_score = batch_update(queue, batch_count, max_avg_score)

def batch_update(queue, batch_count, max_avg_score):
    batch_size=BATCH_SIZE
    dict = {}
    batch_score_sum=0
    for i in range(batch_size):
        episode = queue.get()
        print("Get {}-th episode, Game score: {}".format(i+1, episode[2]))
        batch_score_sum += episode[2]
        updateEvaluation(dict, episode)
    batch_score_avg = batch_score_sum / float(batch_size)
    print("Get {} episodes, start update, Average score: {}".format(batch_size, batch_score_avg))
    with open("result.txt", "a") as f:
        f.write("batch count: {} / score: {}\n".format(batch_count, batch_score_avg))
    print("Size of dict: {}".format(len(dict)))
    for tuple_moved_board in dict.keys():
        moved_board = tuple_to_list(tuple_moved_board)
        delta = dict.get(tuple_moved_board)[0]
        f_handler.updateValue(np.array(moved_board), delta)
    print("---update done, save the latest weights---")
    f_handler.saveWeights("saved_latest_weights.pickle")

    if max_avg_score <= batch_score_avg:
        max_avg_score = batch_score_avg
        print("---Find Maximum Weights and Save it---")
        weight_indices = np.where(f_handler.featureSet[0].getWeight() != 0)
        f_handler.saveWeights("saved_best_weights.pickle")

    del dict
    return max_avg_score


def updateEvaluation(dict, episode):
    board_chain = episode[0]
    reward_chain = episode[1]
    chain_len = len(board_chain)
    lamb = 0.5
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
                weight = 1 - sum_of_weight
                # weight = pow(lamb, j*1.0)
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
    is_all_dead = True
    for index in range(len(players)):
        player = players[index]
        if player.is_alive():
            print("{}-th player is not done.".format(index))
            is_all_dead = False
    return is_all_dead


if __name__ == "__main__":
    # print("before load weight: {}".format(np.where(f_handler.featureSet[0].getWeight() != 0)))
    # print("---Load saved weights---")
    # f_handler.loadWeights("saved_weights.pickle")
    # load_weight_indices = np.where(f_handler.featureSet[0].getWeight() != 0)
    # print("after load weight: {}".format(load_weight_indices))
    # print(f_handler.featureSet[0].getWeight()[load_weight_indices])
    queue = Queue()
    players = []
    updater = Process(target=batch_update_forever, args=(queue, ))
    updater.start()
    while True:
        players = []
        for i in range(PROCESS_NUM):
            player = Process(target=play_game, args=(queue, ))
            players.append(player)
            print("Player start!")
            player.start()
        time.sleep(0.1)
        for player in players:
            player.join()
            print("Process alive? {}".format(player.is_alive()))
            #player.terminate()
            time.sleep(0.1)
        print("Is Game done?: {}".format(isGameDone(players)))
    #while not isGameDone(players):
        #time.sleep(10)
        #print("keep playing")

    # count = 0
    # score_set = []
    # max_score = 0
    # while True:
    #     count += 1
    #     score = TD_Learning()
    #     # print("iter: {} / score: {}".format(count, score))
    #     with open("result.txt", "a") as f:
    #         f.write("iter: {} / score: {}\n".format(count, score))
    #     if score > max_score:
    #         max_score = score
    #         print("---Find Save Maximum Weights---")
    #         weight_indices = np.where(f_handler.featureSet[0].getWeight() != 0)
    #         print("saved weight: {}".format(weight_indices));
    #         print(f_handler.featureSet[0].getWeight()[weight_indices])
    #         f_handler.saveWeights("saved_best_weights.pickle")
    #     score_set.append(score)
    #     #print("final score: {} ".format(score))
    #     f_handler.saveWeights("saved_last_weights.pickle")
    # pygame.quit()
    # print("End Successfully")



