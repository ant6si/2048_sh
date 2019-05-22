import sys
sys.path.append("../")

import numpy as np
from operation import *
from featureSet import *
import pickle

class FeatureHandler(object):
    def __init__(self):
        self.combineMax = combineMaxTileCount()
        self.mergeableTile = mergeableTileCount()
        self.distinctTile = distinctTileCount()
        self.emptyTile = emptyTileCount()
        self.maxTile = maxTileCount()
        self.layerTile = layerTileCount()
        self.axe = axeTuple()
        self.recTangle = recTangTuple()
        self.lineTuple = lineTuple()
        self.featureSet = [self.emptyTile, self.lineTuple, self.recTangle, self.axe, self.maxTile, self.layerTile,
                           self.distinctTile, self.mergeableTile]
        # self.featureSet = [self.emptyTile, self.lineTuple, self.recTangle, self.axe, self.maxTile, self.distinctTile,
        #                    self.mergeableTile]
        #self.featureSet = [self.combineMax]
        #self.featureSet = [self.emptyTile];
        # self.featureSet = []

    def setSymmetricBoards(self, board):
        """

        :param boardStatus: s
        oRows: vertical flip
        reverseRows: horizontal flip
        oReverseRows: vertical + horizontal flip
        rotateBoards: 4X4X4 matrix, (up, left, right, down) boardStatus
        isomorphicBoards: 8X4X4 matirx, (ul, ur, dr, dl, lr, ll, rl, rr) boardStatus
        :return:
        """
        # oRows = np.zeros([4, 4], dtype=np.int)
        # reverseRows = np.zeros([4, 4], dtype=np.int)
        # oReverseRows = np.zeros([4, 4], dtype=np.int)
        #
        # for i in range(4):
        #     rows = getRow(board, i)
        #     oRows[3-i] = rows
        #     reverseRows[i] = reverseRow(rows)
        #     oReverseRows[3-i] = reverseRows[i]

        reverseRows = reverseRow(board)
        oRows = reverseCol(board)
        oReverseRows = reverseRow(reverseCol(board))

        rotateBoards = np.concatenate([board, setCols(reverseRows),
                                       setCols(oRows), oReverseRows], axis=0).reshape([4,4,4])
        isomorphicBoards = np.concatenate([board, reverseRows, oRows, oReverseRows,
                                                setCols(board), setCols(reverseRows),
                                                setCols(oRows), setCols(oReverseRows)]).reshape([8,4,4])
        for i in range(len(self.featureSet)):
            self.featureSet[i].setSymmetricBoards(rotateBoards, isomorphicBoards)

    def getValue(self, board):
        self.setSymmetricBoards(board)
        value = 0
        for idx in range(len(self.featureSet)):
            value += self.featureSet[idx].getScore(board)
        return value

    def updateValue(self, board, delta):
        self.setSymmetricBoards(board)
        part_delta = delta / float(len(self.featureSet))
        for idx in range(len(self.featureSet)):
            self.featureSet[idx].updateScore(board, part_delta)

    def loadWeights(self, weight_file):
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)
        for idx in range(len(self.featureSet)):
            self.featureSet[idx].loadWeight(weights[idx])


    def saveWeights(self, file_name):
        weights = []
        for idx in range(len(self.featureSet)):
            weights.append(self.featureSet[idx].getWeight())
            # save weight to pickle
        with open(file_name, 'wb') as f:
            pickle.dump(weights, f)

if __name__ == "__main__":
    print("hello")

    b = np.array([ [1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4] ])
    delta = 10

    a = FeatureHandler()
    a.getValue(b)
    a.updateValue(b, delta)
    a.getValue(b)


