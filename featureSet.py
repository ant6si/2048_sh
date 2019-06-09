import sys
sys.path.append("../")

import collections
import numpy as np
from operation import *
from feature import feature



class SC_2_Biggest_tile(feature):
    def __init__(self):
        # [dict_1, dict_2, dict_3, dict_4]
        self.big_dict = {}
        self.num_of_tuples_float = 1.0

    def getScore(self, mboard):
        big_key = self.get_big_key(mboard)
        return self.get_key_value(self.big_dict, big_key)


    def updateScore(self, mboard, delta):
        # _delta = delta
        _delta = delta / self.num_of_tuples_float

        big_dict = self.big_dict
        big_key = self.get_big_key(mboard)
        big_value = self.get_key_value(big_dict, big_key)
        big_dict[big_key] = big_value + _delta

    def get_big_key(self, mboard):
        biggest = np.max(mboard)
        big_key = [biggest]

        temp_board = mboard.T
        temp_board2 = temp_board[::-1]
        reverse_board = temp_board2.T

        big_indices = np.where(reverse_board == biggest)
        big_key.append(big_indices[0][big_indices[0].size - 1])
        big_key.append(3 - big_indices[1][big_indices[1].size - 1])

        return tuple(big_key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.big_dict

    def loadWeight(self, weight):
        self.big_dict = weight


class SC_2_Monotonicity(feature):
    def __init__(self):
        # [dict_1, dict_2, dict_3, dict_4]
        mono_dict_0, mono_dict_1, mono_dict_2 = {}, {}, {}

        self.mono_dict_list = [mono_dict_0, mono_dict_1, mono_dict_2]
        self.num_of_tuples_float = 3.0

    def getScore(self, mboard):
        mono_sum = 0
        for idx in range(3):
            idx_dict = self.mono_dict_list[idx]
            idx_key = self.get_mono_key(mboard, idx)
            mono_sum += self.get_key_value(idx_dict, idx_key)  # row_wise
        return mono_sum

    def updateScore(self, mboard, delta):
        # _delta = delta
        _delta = delta / self.num_of_tuples_float
        for idx in range(3):
            idx_dict = self.mono_dict_list[idx]
            idx_key = self.get_mono_key(mboard, idx)
            idx_value = self.get_key_value(idx_dict, idx_key)
            idx_dict[idx_key] = idx_value + _delta


    def get_mono_key(self, mboard, dic_num):

        num_list = []
        if dic_num == 0:
            num_list = np.r_[mboard[3], mboard[2][::-1]]
            # print(num_list)
        if dic_num == 1:
            num_list = np.r_[mboard[2][::-1], mboard[1]]
            # print(num_list)
        if dic_num == 2:
            num_list = np.r_[mboard[1], mboard[0][::-1]]
            # print(num_list)
        mono_list = list()
        mono_list.append(np.max(mboard))
        for idx in range(num_list.size - 1):
            if num_list[idx] > num_list[idx + 1]:
                mono_list.append(1)
            if num_list[idx] == num_list[idx + 1]:
                mono_list.append(0)
            if num_list[idx] < num_list[idx + 1]:
                mono_list.append(-1)

        return tuple(mono_list)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.mono_dict_list

    def loadWeight(self, weight):
        self.mono_dict_list = weight


class SC_Linetuple(feature):
    def __init__(self):
        # [dict_1, dict_2, dict_3, dict_4]
        r_dict_1, r_dict_2, r_dict_3, r_dict_4 = {}, {}, {}, {}
        c_dict_1, c_dict_2, c_dict_3, c_dict_4 = {}, {}, {}, {}
        self.r_dict_list = [r_dict_1, r_dict_2, r_dict_3, r_dict_4]
        self.c_dict_list = [c_dict_1, c_dict_2, c_dict_3, c_dict_4]
        self.tot_dict_list = [self.r_dict_list, self.c_dict_list]
        self.num_of_tuples_float = 8.0

    def getScore(self, mboard):
        line_sum=0;
        for idx in range(4):
            r_dict = self.r_dict_list[idx]
            c_dict = self.c_dict_list[idx]
            line_sum += self.get_key_value(r_dict,tuple(mboard[3-idx])) # row_wise
            line_sum += self.get_key_value(c_dict, tuple(mboard[:, idx])) # col_wise
        return line_sum

    def updateScore(self, mboard, delta):
        # _delta = delta
        _delta = delta / self.num_of_tuples_float
        for idx in range(4):
            r_dict = self.r_dict_list[idx]
            c_dict = self.c_dict_list[idx]
            row_key = tuple(mboard[3-idx])
            col_key = tuple(mboard[:, idx])
            # col_key = tuple(board[:,id])
            r_value = self.get_key_value(r_dict, row_key)
            c_value = self.get_key_value(c_dict, col_key)
            # with open('weight_log.txt', 'a') as f:
            #     f.write("key:{} before update: {} delta: {}".format(row_key, r_dict[row_key], _delta))
            r_dict[row_key] = r_value + _delta
            c_dict[col_key] = c_value + _delta
            # with open("update_log.txt", "a") as f:
            # with open('weight_log.txt', 'a') as f:
            #     f.write("after update: {} \n".format(r_dict[row_key]))

        # print("update dome, line weight 0 length: {}".format(len(self.tot_dict_list[0][0])))

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.tot_dict_list

    def loadWeight(self, weight):
        self.tot_dict_list = weight
        self.r_dict_list = self.tot_dict_list[0]
        self.c_dict_list = self.tot_dict_list[1]

class SC_Rectuple(feature):
    def __init__(self):
        self.rec_dict_list = []
        for idx in range(9):
            self.rec_dict_list.append({})
        self.num_of_tuples_float = 9.0

    def getScore(self, mboard):
        rec_sum=0
        count=-1
        for r in range(3):
            for c in range(3):
                count+=1
                _dict = self.rec_dict_list[count]
                _key = list_to_tuple(mboard[r:r + 2, c:c + 2])
                rec_sum += self.get_key_value(_dict, _key)
        return rec_sum

    def updateScore(self, mboard, delta):
        # _delta = delta
        _delta = delta / self.num_of_tuples_float
        count = -1
        for r in range(3):
            for c in range(3):
                count += 1
                _dict = self.rec_dict_list[count]
                _key = list_to_tuple(mboard[r:r + 2, c:c + 2])
                _v = self.get_key_value(_dict, _key)
                _dict[_key] += _delta

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        _dict[_key] = 0.0 # initialized with 0.1
        return _dict[_key]

    def getWeight(self):
        return self.rec_dict_list

    def loadWeight(self, weight):
        self.rec_dict_list = weight

class HW_lineTuple(feature):
    def __init__(self):
        fourTuple = {}
        self.fourTuple = [fourTuple]

    def getKey(self, board, num):
        if (num != 0 and num != 1):
            print("Wrong Input Number!\n")

        key = tuple(board[:, 3 - num])  # save vector as tuple
        return key

    def updateScore(self, board, delta):
        self.boards = board
        for j in range(2):
            key = self.getKey(self.boards, j)
            symmetricKey = key[::-1]

            ##
            self.fourTuple[0][key] = self.get_key_value(self.fourTuple[0], key) + delta

            if symmetricKey != key:
                    self.fourTuple[0][symmetricKey] = self.get_key_value(self.fourTuple[0], symmetricKey) + delta


    def getScore(self, board):
        self.boards = board
        sum = 0.0
        # 取num=0，1的index
        for j in range(2):
            key = self.getKey(self.boards, j)
            symmetricKey = key[::-1]

            sum += self.get_key_value(self.fourTuple[0], key)
            print(sum)
            if symmetricKey != key:
                sum += self.get_key_value(self.fourTuple[0], symmetricKey)

        return sum

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict.update({_key: 0.0}) # initialized with 0.1
            return _dict[_key]


    def getWeight(self):
        return self.fourTuple

    def loadWeight(self, weight):
        self.fourTuple = weight


class HW_recTangTuple(feature):
    def __init__(self):
        self.sixTuple = {}

    def getKey(self, board, num):
        if (num!=0 and num!=1):
            print("Wrong Input Number!\n")
        k1 = tuple(board[:, num][:-1])
        k2 = tuple(board[:, num+1][:-1])

        key = k1 + k2
        return key

    def updateScore(self, board, delta):
        # print("update recTangTuple")
        self.boards = self.getRotateBoards()

        for i in range(4):
            for j in range(2):
                key1 = self.getKey(self.boards[i], j)
                key2 = self.getKey(reverseRow(self.boards[i]),j)
                if key1==key2 and j==1:
                    self.sixTuple[key1] = self.get_key_value(self.sixTuple, key1) + delta
                else:
                    self.sixTuple[key1] = self.get_key_value(self.sixTuple, key1) + delta
                    self.sixTuple[key2] = self.get_key_value(self.sixTuple, key2) + delta
        # print("update recTangTuple Done")

    def getScore(self, board):
        self.boards = self.getRotateBoards()

        sum = 0.0
        for i in range(4):
            key1 = self.getKey(self.boards[i], 0)
            key2 = self.getKey(self.boards[i], 1)
            key3 = self.getKey(reverseRow(self.boards[i]), 0)

            sum += self.get_key_value(self.sixTuple, key1)
            sum += self.get_key_value(self.sixTuple, key2)
            sum += self.get_key_value(self.sixTuple, key3)
        return sum

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.sixTuple

    def loadWeight(self, weight):
        self.sixTuple = weight


class HW_axeTuple(feature):
    def __init__(self):
        self.sixTuple = {}

    def getKey(self, board, num):
        if num>=3:
            print("Wrong Input Number!\n")
        k1 = tuple(board[:, num])
        k2 = tuple(board[:, num+1][-2:])
        key = k1 + k2
        return key

    def updateScore(self, board, delta):
        # print("update axeTuple")

        self.boards = self.getRotateBoards()
        for i in range(4):
            for j in range(3):
                key1 = self.getKey(self.boards[i], j)
                key2 = self.getKey(reverseRow(self.boards[i]),j)

                self.sixTuple[key1] = self.get_key_value(self.sixTuple, key1) + delta
                self.sixTuple[key2] = self.get_key_value(self.sixTuple, key2) + delta
        # print("update axeTuple Done")

    def getScore(self, board):
        self.boards = self.getRotateBoards()

        sum = 0.0
        for i in range(4):
            for j in range(3):
                key1 = self.getKey(self.boards[i],j)
                key2 = self.getKey(reverseRow(self.boards[i]), j)

                sum += self.get_key_value(self.sixTuple, key1)
                sum += self.get_key_value(self.sixTuple, key2)
        return sum

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.sixTuple;

    def loadWeight(self, weight):
        self.sixTuple = weight

class HW_maxTileCount(feature):
    def __init__(self):
        self.maxTile = {}

    def getKey(self, board, num=0):
        board = tuple(board.reshape([1, -1]).tolist())
        key = 0
        for i in range(10, 16):
            key += board.count(2 ** i)
        return key

    def updateScore(self, board, delta):
        # print("update MaxTileCount")
        key = self.getKey(board, 0)
        self.maxTile[key] = self.get_key_value(self.maxTile, key) + delta
        # print("update MaxTileCount Done")

    def getScore(self, board):
        key = self.getKey(board, 0)
        return self.get_key_value(self.maxTile, key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.maxTile

    def loadWeight(self, weight):
        self.maxTile = weight

class HW_emptyTileCount(feature):
    def __init__(self):
        self.emptyTile = {}

    # 返回0个数
    def getKey(self, board, num=0):
        board = tuple(board.reshape([1, -1]).tolist())
        key = board.count(0)
        return key

    # 每次update单纯的累计delta
    def updateScore(self, board, delta):
        # print("update emptyTileCount")
        key = self.getKey(board, 0)
        self.emptyTile[key] = self.get_key_value(self.emptyTile, key) + delta
       # print("update emptyTileCount done")

    # e.g. 返回有2个0 对应的weight
    def getScore(self, board):
        key = self.getKey(board, 0)
        return self.get_key_value(self.emptyTile, key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0  # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.emptyTile

    def loadWeight(self, weight):
        self.emptyTile = weight

class HW_mergeableTileCount(feature):
    def __init__(self):
        self.mergeableTile= {}

    def getKey(self, board, num):
        key = 0
        for i in range(3):
            key += np.sum(board[i]==board[i+1])
            key += np.sum(board[:, i]==board[:, i+1])
        return key

    def updateScore(self, board, delta):
        # print("update mergeableTileCount")
        key = self.getKey(board, 0)
        self.mergeableTile[key] = self.get_key_value(self.mergeableTile, key) + delta
        # print("update mergeableTileCount done")

    def getScore(self, board):
        key = self.getKey(board, 0)
        return self.get_key_value(self.mergeableTile, key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0  # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.mergeableTile

    def loadWeight(self, weight):
        self.mergeableTile = weight

class HW_distinctTileCount(feature):
    def __init__(self):
        self.distinctTile = {}

    def getKey(self, board, num):
        # bitset = 0
        # for i in range(16):
        #     bitset = bitset and board[-i]
        #
        # bitset >> 1
        # count = 0
        # while (bitset):
        #     bitset = bitset and (bitset - 1)
        #     count += 1

        distinctTile = len(set(board.reshape([-1]).tolist()))
        if not (distinctTile>=0 and distinctTile<=15):
            print(distinctTile, '\n')
            return 0
        return distinctTile

    def updateScore(self, board, delta):
        # print("update distinctTileCount")
        key = self.getKey(board, 0)
        self.distinctTile[key] = self.get_key_value(self.distinctTile, key) + delta
        # print("update distinctTileCount done")

    def getScore(self, board):
        key = self.getKey(board, 0)
        return self.get_key_value(self.distinctTile, key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.distinctTile

    def loadWeight(self, weight):
        self.distinctTile = weight

class HW_layerTileCount(feature):
    def __init__(self):
        self.layerTile = {}

    def getKey(self, board, num):
        index = 0
        # for i in range(3):
        #     for j in range(3):
        #         if board[i][j]==2*board[i][j+1] or 2*board[i][j]==board[i][j+1]:
        #             index += 1
        #         if board[i][j]==2*board[i+1][j] or 2*board[i][j]==board[i+1][j]:
        #             index += 1
        board = np.where(board==0, -1, board)
        for i in range(3):
            index += np.sum(getRow(board, i) == 2*getRow(board, i + 1)) + \
                     np.sum(2*getRow(board, i) == getRow(board, i + 1))
            index += np.sum(getCol(board, i) == 2*getCol(board, i + 1)) + \
                     np.sum(2*getCol(board, i) == getCol(board, i + 1))
        return index

    def updateScore(self, board, delta):
        # print("update layerTileCount")
        key = self.getKey(board, 0)
        self.layerTile[key] = self.get_key_value(self.layerTile, key) + delta
        # print("update layerTileCount done")

    def getScore(self, board):
        key = self.getKey(board, 0)
        return self.get_key_value(self.layerTile, key)

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        return self.layerTileCount

    def loadWeight(self, weight):
        self.layerTile = weight

if __name__=='__main__':
    pass
