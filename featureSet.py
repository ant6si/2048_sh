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

class lineTuple(feature):
    def __init__(self):
        self.fourTuple = np.zeros([2,16,16,16,16], dtype=np.float)

    # 取2或3列
    def getIndex(self, board, num):
        if (num!=0 and num!=1):
            print("Wrong Input Number!\n")

        col = getCol(board, 3-num)
        # index = np.zeros([4], dtype=int)
        # for i in range(4):
        #     if col[i]!=0:
        #         index[i] = np.log2(col[i]).astype(int)
        index = np.where(col!= 0, np.log2(col), 0).astype(int)
        return index

    def getSymmetricIndex(self, index):
        return reverseRow(index)

    def updateScore(self, board, delta):
        # print("update LineTuple")
        self.boards = self.getRotateBoards()

        for i in range(4):
            for j in range(2):
                index = self.getIndex(self.boards[i], j)
                symmetricIndex = reverseRow(index)

                self.fourTuple[j][index[0]][index[1]][index[2]][index[3]] += delta
                if np.array_equal(symmetricIndex, index):
                    self.fourTuple[j][symmetricIndex[0]][symmetricIndex[1]][symmetricIndex[2]][symmetricIndex[3]] += delta
        # print("update LineTuple Done")


    # 4个横，4个列的score的sum， 当symmetric不一样的话要再加
    def changeToOx(self, np):
        # np = hex(np)
        # idx = np[0]<<3 | np[1]<<2 | np[2]<<1 | np[3]
        idx = np[0]*16**3 + np[1]*16**2 + np[2]*16**1 + np[3]
        return idx

    def getScore(self, board):
        self.boards = self.getRotateBoards() #? rotate 가 안되는 것 같은데요?
        # self.boards = board

        sum = 0.0
        for i in range(4):
            # 取num=0，1的index
            for j in range(2):
                index = self.getIndex(self.boards[i],j)
                symmetricIndex = reverseRow(index)

                sum += self.fourTuple[j][index[0]][index[1]][index[2]][index[3]]

                if np.array_equal(symmetricIndex, index):
                    sum += self.fourTuple[j][symmetricIndex[0]][symmetricIndex[1]][symmetricIndex[2]][symmetricIndex[3]]
        # idx, sIdx = [], []
        # for i in range(4):
        #     for j in range(2):
        #         index = self.getIndex(self.boards, j)
        #         symmetricIndex = self.getSymmetricIndex(index)
        #         idx.append(self.changeToOx(index))
        #         sIdx.append(self.changeToOx(symmetricIndex))
        # print(idx)
        # self.fourTuple[1][14271] = 1
        # sum += self.fourTuple[j][idx]
        # print("sum", sum)
        return sum


    def getWeight(self):
        return self.fourTuple

    def loadWeight(self, weight):
        self.fourTuple = weight

class recTangTuple(feature):
    def __init__(self):
        self.sixTuple = np.zeros([2,16,16,16,16,16,16], dtype=np.float)

    def getIndex(self, board, num):
        if (num!=0 and num!=1):
            print("Wrong Input Number!\n")
        c1 = getCol(board, num)[:-1]
        c2 = getCol(board, num+1)[:-1]

        col = np.concatenate([c1, c2], axis=0)
        index = np.where(col != 0, np.log2(col), 0).astype(int)
        return index

    def updateScore(self, board, delta):
        # print("update recTangTuple")
        self.boards = self.getRotateBoards()

        for i in range(4):
            for j in range(2):
                index1 = self.getIndex(self.boards[i], j)
                index2 = self.getIndex(reverseRow(self.boards[i]),j)
                if np.array_equal(index1, index2) and j==1:
                    self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]] += delta
                else:
                    self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]] += delta
                    self.sixTuple[j][index2[0]][index2[1]][index2[2]][index2[3]][index2[4]][index2[5]] += delta
        # print("update recTangTuple Done")


    def getScore(self, board):
        self.boards = self.getRotateBoards()

        sum = 0.0
        # for i in range(4):
        #     for j in range(2):
        #         index1 = self.getIndex(self.boards,j)
        #         index2 = self.getIndex(reverseRow(self.boards), j)
        #
        #         if (j==1):
        #             sum += self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]]
        #         else:
        #             sum += self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]]
        #             sum += self.sixTuple[j][index2[0]][index2[1]][index2[2]][index2[3]][index2[4]][index2[5]]
        for i in range(4):
            index1 = self.getIndex(self.boards[i], 0)
            index2 = self.getIndex(self.boards[i], 1)
            index3 = self.getIndex(reverseRow(self.boards[i]), 0)

            sum += self.sixTuple[0][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]]
            sum += self.sixTuple[1][index2[0]][index2[1]][index2[2]][index2[3]][index2[4]][index2[5]]
            sum += self.sixTuple[0][index3[0]][index3[1]][index3[2]][index3[3]][index3[4]][index3[5]]

        return sum

    def getWeight(self):
        return self.sixTuple

    def loadWeight(self, weight):
        self.sixTuple = weight

class axeTuple(feature):
    def __init__(self):
        self.sixTuple = np.zeros([3,16,16,16,16,16,16], dtype=np.float)

    def getIndex(self, board, num):
        if num>=3:
            print("Wrong Input Number!\n")
        c1 = getCol(board, num)
        c2 = getCol(board, num+1)[-2:]

        col = np.concatenate([c1, c2], axis=0)
        index = np.where(col != 0, np.log2(col), 0).astype(int)
        return index

    def updateScore(self, board, delta):
        # print("update axeTuple")

        self.boards = self.getRotateBoards()
        for i in range(4):
            for j in range(3):
                index1 = self.getIndex(self.boards[i], j)
                index2 = self.getIndex(reverseRow(self.boards[i]),j)
                self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]] += delta
                self.sixTuple[j][index2[0]][index2[1]][index2[2]][index2[3]][index2[4]][index2[5]] += delta
        # print("update axeTuple Done")

    def getScore(self, boardStatus):
        self.boards = self.getRotateBoards()

        sum = 0.0
        for i in range(4):
            for j in range(3):
                index1 = self.getIndex(self.boards[i],j)
                index2 = self.getIndex(reverseRow(self.boards[i]), j)

                sum += self.sixTuple[j][index1[0]][index1[1]][index1[2]][index1[3]][index1[4]][index1[5]]
                sum += self.sixTuple[j][index2[0]][index2[1]][index2[2]][index2[3]][index2[4]][index2[5]]
        return sum

    def getWeight(self):
        return self.sixTuple;

    def loadWeight(self, weight):
        self.sixTuple = weight

class combineMaxTileCount(feature):
    def __init__(self):
        self.combineMaxTile = np.zeros([16**6], dtype=np.float)

    def getIndex(self, board, num):
        index = 0
        # for i in range(16):
        #     val = board[i]
        #     if val>=11:
        #         shift = val - 11
        #         index += 2^(2*shift+14)
        #     if val==0:
        #         index += 2^10
        #     if ((60-i)/4)%4!=3:
        #         right = board[i-5:i-4]
        #         if val!=0 and val==right:
        #             index += 2^5
        #         if (val-right)==1 or (val-right)==-1:
        #             index += 1
        #     if ((60-i)/16)%4 != 3:
        #         down = board[i-17:i-16]
        #         if val!=0 and val==down:
        #             index += 2^5
        #         if (val-down)==1 or (val-down)==-1:
        #             index += 1
        index = 0
        logValue = np.where(board != 0, np.log2(board), 0).astype(int)

        # maxTile
        highValue = np.where(logValue >= 11, logValue - 11, -1)
        counts = np.array([np.count_nonzero(highValue == i) for i in range(6)])
        base = np.array([2 ** i for i in range(6)])
        index += np.sum(np.multiply(2 ** (2 * base), counts))

        # emptyTileCount
        numOfIndex = np.where(board == num)[0]
        index += len(numOfIndex)

        for i in range(3):

            # mergeableTileCount
            index += np.sum(getRow(board, i) == getRow(board, i + 1))
            index += np.sum(getCol(board, i) == getCol(board, i + 1))

            # layerTileCount
            index += np.sum(getRow(board, i) == 2 * getRow(board, i + 1)) + \
                     np.sum(2 * getRow(board, i) == getRow(board, i + 1))
            index += np.sum(getCol(board, i) == 2 * getCol(board, i + 1)) + \
                     np.sum(2 * getCol(board, i) == getCol(board, i + 1))

        return index

    def updateScore(self, board, delta):
        # print("update combineMaxTileCount")
        self.combineMaxTile[self.getIndex(board, 0)] += delta
        # print("update combineMaxTileCount Done")


    def getScore(self, board):
        return self.combineMaxTile[self.getIndex(board, 0)]

    def getWeight(self):
        return self.combineMaxTile

    def loadWeight(self, weight):
        self.combineMaxTile = weight

class maxTileCount(feature):
    def __init__(self):
        self.maxTile = np.zeros([512], dtype=np.float)

    def getIndex(self, board, num):
        logValue = np.where(board!=0, np.log2(board), 0).astype(int)
        highValue= np.where(logValue >= 11, logValue - 11, -1)
        counts = np.array([np.count_nonzero(highValue == i) for i in range(6)])
        base = np.array([2**i for i in range(6)])
        index = np.sum(np.multiply(2**(2*base), counts))
        return index

    def updateScore(self, board, delta):
        # print("update MaxTileCount")
        self.maxTile[self.getIndex(board, 0)] += delta
        # print("update MaxTileCount Done")


    def getScore(self, board):
        return self.maxTile[self.getIndex(board, 0)]

    def getWeight(self):
        return self.maxTile

    def loadWeight(self, weight):
        self.maxTile = weight

class emptyTileCount(feature):
    def __init__(self):
        self.emptyTile = np.zeros([16], dtype=np.float)

    # 返回0个数
    def getIndex(self, board, num=0):
        numOfIndex = np.where(board==num)[0]
        index = len(numOfIndex)
        return index

    # 每次update单纯的累计delta
    def updateScore(self, board, delta):
        # print("update emptyTileCount")
        self.emptyTile[self.getIndex(board, 0)] += delta
        # print("update emptyTileCount done")

    # e.g. 返回有2个0 对应的weight
    def getScore(self, board):
        return self.emptyTile[self.getIndex(board, 0)]

    def getWeight(self):
        return self.emptyTile

    def loadWeight(self, weight):
        self.emptyTile = weight

class mergeableTileCount(feature):
    def __init__(self):
        self.mergeableTile= np.array(range(24), dtype=np.float)

    def getIndex(self, board, num):
        index = 0
        # for i in range(4):
        #     row = board[i]
        #     col = getCol(board, i)
        #     for j in range(3):
        #         if row[j]==row[j+1]:
        #             index += 1

        #         if col[j]==col[j+1]:
        #             index += 1
        for i in range(3):
            index += np.sum(getRow(board, i)==getRow(board, i+1))
            index += np.sum(getCol(board, i)==getCol(board, i+1))
        return index

    def updateScore(self, board, delta):
        # print("update mergeableTileCount")
        self.mergeableTile[self.getIndex(board, 0)] += delta
        # print("update mergeableTileCount done")

    def getScore(self, board):
        return self.mergeableTile[self.getIndex(board, 0)]

    def getWeight(self):
        return self.mergeableTile

    def loadWeight(self, weight):
        self.mergeableTile = weight

class distinctTileCount(feature):
    def __init__(self):
        self.distinctTile= np.zeros([16], dtype=np.float)

    def getIndex(self, board, num):
        # bitset = 0
        # for i in range(16):
        #     bitset = bitset and board[-i]
        #
        # bitset >> 1
        # count = 0
        # while (bitset):
        #     bitset = bitset and (bitset - 1)
        #     count += 1
        counts = collections.Counter(board.reshape([-1]).tolist())
        distinctTile = len(counts.keys()) - 1 # except zero

        if not (distinctTile>=0 and distinctTile<=15):
            print(distinctTile, '\n')
            return 0
        return distinctTile

    def updateScore(self, board, delta):
        # print("update distinctTileCount")
        self.distinctTile[self.getIndex(board, 0)] += delta
        # print("update distinctTileCount done")

    def getScore(self, board):
        return self.distinctTile[self.getIndex(board, 0)]

    def getWeight(self):
        return self.distinctTile

    def loadWeight(self, weight):
        self.distinctTile = weight

class layerTileCount(feature):
    def __init__(self):
        self.layerTile = np.zeros([128], dtype=np.float)

    def getIndex(self, board, num):
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
        self.layerTile[self.getIndex(board, 0)] += delta
        # print("update layerTileCount done")

    def getScore(self, board):
        return self.layerTile[self.getIndex(board, 0)]

    def getWeight(self):
        return layerTileCount

    def loadWeight(self, weight):
        self.layerTile = weight


class HW_lineTuple(feature):
    def __init__(self):
        self.fourTuple = {}

    def getKey(self, board, num):
        if (num != 0 and num != 1):
            print("Wrong Input Number!\n")

        key = tuple(board[:, 3 - num])  # save vector as tuple
        return key

    def updateScore(self, board, delta):
        # print("update LineTuple")
        self.boards = self.getRotateBoards()  # get 4 boards

        for i in range(4):
            for j in range(2):
                key = self.getKey(self.boards[i], j)
                symmetricKey = key[::-1]

                ##
                self.fourTuple[key] = self.get_key_value(self.fourTuple, key) + delta

                if symmetricKey != key:
                    self.fourTuple[symmetricKey] = self.get_key_value(self.fourTuple, symmetricKey) + delta

    def getScore(self, board):
        self.boards = self.getRotateBoards()  # ? rotate 가 안되는 것 같은데요?

        sum = 0.0
        for i in range(4):
            # 取num=0，1的index
            for j in range(2):
                key = self.getKey(self.boards[i], j)
                symmetricKey = key[::-1]

                sum += self.get_key_value(self.fourTuple, key)

                if symmetricKey != key:
                    sum += self.get_key_value(self.fourTuple, symmetricKey)
        return sum

    def get_key_value(self, _dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            _dict[_key] = 0.0 # initialized with 0.1
            return _dict[_key]


    def getWeight(self):
        return [self.fourTuple]

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
        return layerTileCount

    def loadWeight(self, weight):
        self.layerTile = weight

if __name__=='__main__':
    pass
