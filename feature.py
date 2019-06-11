import sys
sys.path.append("../")

import numpy as np
from operation import *


class feature():
    def __init__(self):
        pass

    def getKey(self, board, num):
        pass

    def updateScore(self, board, delta):
        pass

    def getScore(self, board):
        pass

    def setSymmetricBoards(self, rotateSymmetry, isomorphic):
        """
        :param rotateSymmetry: including (up, down, letf, right) 4 boards
        :param isomorphic: including rotateSymmetry board and its mirrorsymmetric board, total 8 boards
        """
        self.rotateBoards = rotateSymmetry
        self.isomorphicBoards = isomorphic

    def getRotateBoards(self):
        """
        :return: rotatedSymmetry board
        """
        return self.rotateBoards

    # horizontal symmetric
    def getMirrorSymmetricBoard(self, board):
        """
        :param board: board state
        :return: mirror symmetric board
        """
        reverseRows = reverseRow(board)
        return reverseRows
