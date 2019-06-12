import sys
sys.path.append("../")
import os.path

import numpy as np
from operation import *
from featureSet import *
import pickle

class FeatureHandler(object):
    # def __init__(self, args):
    def __init__(self):
        """
        initialize all feature set
        """
        self.SC_LineTuples = SC_Linetuple()
        self.SC_RecTuples = SC_Rectuple()
        self.SC_2_mono = SC_2_Monotonicity()
        self.SC_2_big = SC_2_Biggest_tile()


        self.HW_linetuple = HW_lineTuple()
        self.HW_rectuple = HW_recTangTuple()
        self.HW_axetuple = HW_axeTuple()
        self.HW_max = HW_maxTileCount()
        self.HW_mergeable = HW_mergeableTileCount()
        self.HW_layer = HW_layerTileCount()
        self.HW_distinct = HW_distinctTileCount()
        self.HW_empty = HW_emptyTileCount()

        # self.featureSet = [self.SC_LineTuples, self.SC_RecTuples]  # basic, comb 1
        # self.featureSet = [self.HW_max, self.HW_mergeable, self.HW_layer, self.HW_distinct, self.HW_empty]  #simple, comb 2
        self.featureSet = [self.SC_2_big, self.SC_2_mono, self.HW_mergeable, self.HW_layer, self.HW_distinct, self.HW_empty]  #our proposed, comb3


    def setSymmetricBoards(self, board):
        """
        :param board: s
        oRows: vertical flip
        reverseRows: horizontal flip
        oReverseRows: vertical + horizontal flip
        rotateBoards: 4X4X4 matrix, (up, left, right, down) boardStatus
        isomorphicBoards: 8X4X4 matirx, (ul, ur, dr, dl, lr, ll, rl, rr) boardStatus
        """
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
        """
        return the state value of board
        """
        #self.setSymmetricBoards(board)
        value = 0
        for idx in range(len(self.featureSet)):
            value += self.featureSet[idx].getScore(board)
        return value

    def get_detail_value(self, board):
        """
        return the state value of board for each features
        """
        #self.setSymmetricBoards(board)
        value_list = []
        for idx in range(len(self.featureSet)):
            value_list.append(self.featureSet[idx].getScore(board))
        return value_list

    def updateValue(self, board, delta):
        """
        update the state value of board
        """
        # self.setSymmetricBoards(board)
        part_delta = delta / float(len(self.featureSet))
        for idx in range(len(self.featureSet)):
            self.featureSet[idx].updateScore(board, part_delta)
            # self.featureSet[idx].updateScore(board, delta)

    def loadWeights(self, weight_file):
        """
        read file and load the weights of feature set
        """
        if os.path.exists(weight_file):
            with open(weight_file, 'rb') as f:
                weights = pickle.load(f)
            for idx in range(len(self.featureSet)):
                self.featureSet[idx].loadWeight(weights[idx])
        else:
            print("File Not Exists, make new weight_file")


    def saveWeights(self, file_name):
        """
        save weight as a pickle file
        """
        weights = []
        for idx in range(len(self.featureSet)):
            weights.append(self.featureSet[idx].getWeight())
            # save weight to pickle
        with open(file_name, 'wb') as f:
            pickle.dump(weights, f)
