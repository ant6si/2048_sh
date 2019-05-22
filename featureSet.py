import sys
sys.path.append("../")

import collections
import numpy as np
from operation import *
from feature import feature

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
        self.boards = self.getRotateBoards()
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


if __name__=='__main__':
    rec = recTangTuple()
    board = np.array([2**i for i in range(16)]).reshape([4,4])
    print("board:")
    print(board)
    index = rec.getIndex(board,1)
    print("index:")
    print(index)
    score = rec.getScore(board)
    print("score:")
    # print(score)