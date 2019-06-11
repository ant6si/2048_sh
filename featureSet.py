import sys
sys.path.append("../")

import collections
import numpy as np
from operation import *
from feature import feature


class SC_Linetuple(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        # [dict_1, dict_2, dict_3, dict_4]
        r_dict_1, r_dict_2, r_dict_3, r_dict_4 = {}, {}, {}, {}
        c_dict_1, c_dict_2, c_dict_3, c_dict_4 = {}, {}, {}, {}
        self.r_dict_list = [r_dict_1, r_dict_2, r_dict_3, r_dict_4]
        self.c_dict_list = [c_dict_1, c_dict_2, c_dict_3, c_dict_4]
        self.tot_dict_list = [self.r_dict_list, self.c_dict_list]
        self.num_of_tuples_float = 8.0

    def getScore(self, mboard):
        """
        return the state value of input board
        """
        line_sum=0;
        for idx in range(4):
            r_dict = self.r_dict_list[idx]
            c_dict = self.c_dict_list[idx]
            line_sum += self.get_key_value(r_dict,tuple(mboard[3-idx])) # row_wise
            line_sum += self.get_key_value(c_dict, tuple(mboard[:, idx])) # col_wise
        return line_sum

    def updateScore(self, mboard, delta):
        """
        update the state value of input board
        """
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
        """
        return weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return weights dictionary
        """
        return self.tot_dict_list

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.tot_dict_list = weight
        self.r_dict_list = self.tot_dict_list[0]
        self.c_dict_list = self.tot_dict_list[1]

class SC_Rectuple(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        self.rec_dict_list = []
        for idx in range(9):
            self.rec_dict_list.append({})
        self.num_of_tuples_float = 9.0

    def getScore(self, mboard):
        """
        return the state value of input board
        """
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
        """
        update the state value of input board
        """
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
        """
        return weight
        """
        if _key in _dict:
            return _dict[_key]
        _dict[_key] = 0.0 # initialized with 0.1
        return _dict[_key]

    def getWeight(self):
        """
        return weights dictionary
        """
        return self.rec_dict_list

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.rec_dict_list = weight



class SC_2_Biggest_tile(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        # [dict_1, dict_2, dict_3, dict_4]
        self.big_dict = {}
        self.num_of_tuples_float = 1.0

    def getScore(self, mboard):
        """
        return the state value of input board
        """
        big_key = self.get_big_key(mboard)
        return self.get_key_value(self.big_dict, big_key)


    def updateScore(self, mboard, delta):
        """
        update the state value of input board
        """
        # _delta = delta
        _delta = delta / self.num_of_tuples_float

        big_dict = self.big_dict
        big_key = self.get_big_key(mboard)
        big_value = self.get_key_value(big_dict, big_key)
        big_dict[big_key] = big_value + _delta

    def get_big_key(self, mboard):
        """
        find biggest grid, and measure the distance from leftdown corner
        return: (biggest number, row distance, column distance)
        """
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
        """
        return weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return weights dictionary
        """
        return self.big_dict

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.big_dict = weight


class SC_2_Monotonicity(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        # [dict_1, dict_2, dict_3, dict_4]
        mono_dict_0, mono_dict_1, mono_dict_2 = {}, {}, {}

        self.mono_dict_list = [mono_dict_0, mono_dict_1, mono_dict_2]
        self.num_of_tuples_float = 3.0

    def getScore(self, mboard):
        """
        return the state value of input board
        """
        mono_sum = 0
        for idx in range(3):
            idx_dict = self.mono_dict_list[idx]
            idx_key = self.get_mono_key(mboard, idx)
            mono_sum += self.get_key_value(idx_dict, idx_key)  # row_wise
        return mono_sum

    def updateScore(self, mboard, delta):
        """
        update the state value of input board
        """
        # _delta = delta
        _delta = delta / self.num_of_tuples_float
        for idx in range(3):
            idx_dict = self.mono_dict_list[idx]
            idx_key = self.get_mono_key(mboard, idx)
            idx_value = self.get_key_value(idx_dict, idx_key)
            idx_dict[idx_key] = idx_value + _delta


    def get_mono_key(self, mboard, dic_num):
        """
        quantify monotonicity of rows
        :return quantified two neighboring rows
        """
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
        """
        return weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return weigths dictionary
        """
        return self.mono_dict_list

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.mono_dict_list = weight


class HW_lineTuple(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        fourTuple={}
        self.fourTuple = [fourTuple]

    def getKey(self, board, num):
        """
        make column values as a key of look-up table
        """
        if (num != 0 and num != 1):
            print("Wrong Input Number!\n")

        key = tuple(board[:, 3 - num])  # save vector as tuple
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("dict type2: {}".format(type(self.fourTuple[0])))
        self.boards = board

        for j in range(2):
            key = self.getKey(self.boards, j)
            symmetricKey = key[::-1]

            ##
            self.fourTuple[0][key] = self.get_key_value(self.fourTuple[0], key) + delta

            if symmetricKey != key:
                self.fourTuple[0][symmetricKey] = self.get_key_value(self.fourTuple[0], symmetricKey) + delta

    def getScore(self, board):
        """
        return: the state value of input board
        """
        # print("dict type1: {}".format(type(self.fourTuple[0])))

        self.boards = board
        sum = 0.0

        for j in range(2):
            key = self.getKey(self.boards, j)
            symmetricKey = key[::-1]
            sum += self.get_key_value(self.fourTuple[0], key)
            if symmetricKey != key:
                sum += self.get_key_value(self.fourTuple[0], symmetricKey)
        return sum

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict.update({_key: 0.0})
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.fourTuple

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.fourTuple = weight


class HW_recTangTuple(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        sixTuple = {}
        self.sixTuple = [sixTuple]

    def getKey(self, board, num):
        """
        make column values as a key of look-up table
        """
        if (num!=0 and num!=1):
            print("Wrong Input Number!\n")
        k1 = tuple(board[:, num][:-1])
        k2 = tuple(board[:, num+1][:-1])

        key = k1 + k2
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update recTangTuple")
        self.boards = board

        for j in range(2):
            key1 = self.getKey(self.boards, j)
            key2 = self.getKey(reverseRow(self.boards),j)
            if key1==key2 and j==1:
                self.sixTuple[0][key1] = self.get_key_value(self.sixTuple[0], key1) + delta
            else:
                self.sixTuple[0][key1] = self.get_key_value(self.sixTuple[0], key1) + delta
                self.sixTuple[0][key2] = self.get_key_value(self.sixTuple[0], key2) + delta
        # print("update recTangTuple Done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        self.boards = board

        sum = 0.0
        key1 = self.getKey(self.boards, 0)
        key2 = self.getKey(self.boards, 1)
        key3 = self.getKey(reverseRow(self.boards), 0)

        sum += self.get_key_value(self.sixTuple[0], key1)
        sum += self.get_key_value(self.sixTuple[0], key2)
        sum += self.get_key_value(self.sixTuple[0], key3)
        return sum

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.sixTuple

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.sixTuple = weight


class HW_axeTuple(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        sixTuple = {}
        self.sixTuple = [sixTuple]

    def getKey(self, board, num):
        """
        make column values as a key of look-up table
        """
        if num>=3:
            print("Wrong Input Number!\n")
        k1 = tuple(board[:, num])
        k2 = tuple(board[:, num+1][-2:])
        key = k1 + k2
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update axeTuple")
        self.boards = board

        for j in range(3):
            key1 = self.getKey(self.boards, j)
            key2 = self.getKey(reverseRow(self.boards),j)

            self.sixTuple[0][key1] = self.get_key_value(self.sixTuple[0], key1) + delta
            self.sixTuple[0][key2] = self.get_key_value(self.sixTuple[0], key2) + delta
        # print("update axeTuple Done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        self.boards = board

        sum = 0.0
        for j in range(3):
            key1 = self.getKey(self.boards,j)
            key2 = self.getKey(reverseRow(self.boards), j)

            sum += self.get_key_value(self.sixTuple[0], key1)
            sum += self.get_key_value(self.sixTuple[0], key2)
        return sum

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.sixTuple;

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.sixTuple = weight


class HW_maxTileCount(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        maxTile = {}
        self.maxTile = [maxTile]

    def getKey(self, board, num=0):
        """
        make column values as a key of look-up table
        """
        board = tuple(board.reshape([1, -1]).tolist())
        key = 0
        for i in range(10, 16):
            key += board.count(2 ** i)
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update MaxTileCount")
        key = self.getKey(board, 0)
        self.maxTile[0][key] = self.get_key_value(self.maxTile[0], key) + delta
        # print("update MaxTileCount Done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        key = self.getKey(board, 0)
        return self.get_key_value(self.maxTile[0], key)

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.maxTile

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.maxTile = weight


class HW_emptyTileCount(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        emptyTile = {}
        self.emptyTile = [emptyTile]

    def getKey(self, board, num=0):
        """
        make column values as a key of look-up table
        """
        board = tuple(board.reshape([1, -1]).tolist())
        key = board.count(0)
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update emptyTileCount")
        key = self.getKey(board, 0)
        self.emptyTile[0][key] = self.get_key_value(self.emptyTile[0], key) + delta
       # print("update emptyTileCount done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        key = self.getKey(board, 0)
        return self.get_key_value(self.emptyTile[0], key)

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0  # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.emptyTile

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.emptyTile = weight

class HW_mergeableTileCount(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        mergeableTile = {}
        self.mergeableTile= [mergeableTile]

    def getKey(self, board, num):

        key = 0
        for i in range(3):
            key += np.sum(board[i]==board[i+1])
            key += np.sum(board[:, i]==board[:, i+1])
        return key

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update mergeableTileCount")
        key = self.getKey(board, 0)
        self.mergeableTile[0][key] = self.get_key_value(self.mergeableTile[0], key) + delta
        # print("update mergeableTileCount done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        key = self.getKey(board, 0)
        return self.get_key_value(self.mergeableTile[0], key)

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0  # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.mergeableTile

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.mergeableTile = weight

class HW_distinctTileCount(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        distinctTile = {}
        self.distinctTile = [distinctTile]

    def getKey(self, board, num):
        """
        make column values as a key of look-up table
        """

        distinctTile = len(set(board.reshape([-1]).tolist()))
        if not (distinctTile>=0 and distinctTile<=15):
            print(distinctTile, '\n')
            return 0
        return distinctTile

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update distinctTileCount")
        key = self.getKey(board, 0)
        self.distinctTile[0][key] = self.get_key_value(self.distinctTile[0], key) + delta
        # print("update distinctTileCount done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        key = self.getKey(board, 0)
        return self.get_key_value(self.distinctTile[0], key)

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.distinctTile

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.distinctTile = weight

class HW_layerTileCount(feature):
    def __init__(self):
        """
        initialize weight as dictionary format
        """
        layerTile = {}
        self.layerTile = [layerTile]

    def getKey(self, board, num):
        """
        make column values as a key of look-up table
        """
        index = 0
        board = np.where(board==0, -1, board)
        for i in range(3):
            index += np.sum(getRow(board, i) == 2*getRow(board, i + 1)) + \
                     np.sum(2*getRow(board, i) == getRow(board, i + 1))
            index += np.sum(getCol(board, i) == 2*getCol(board, i + 1)) + \
                     np.sum(2*getCol(board, i) == getCol(board, i + 1))
        return index

    def updateScore(self, board, delta):
        """
        update the state value of input board
        """
        # print("update layerTileCount")
        key = self.getKey(board, 0)
        self.layerTile[0][key] = self.get_key_value(self.layerTile[0], key) + delta
        # print("update layerTileCount done")

    def getScore(self, board):
        """
        return: the state value of input board
        """
        key = self.getKey(board, 0)
        return self.get_key_value(self.layerTile[0], key)

    def get_key_value(self, _dict, _key):
        """
        return: weight
        """
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]

    def getWeight(self):
        """
        return: weights dictionary
        """
        return self.layerTile

    def loadWeight(self, weight):
        """
        load weight as dictionary
        """
        self.layerTile = weight
