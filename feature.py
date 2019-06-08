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
        self.rotateBoards = rotateSymmetry
        self.isomorphicBoards = isomorphic

    def getRotateBoards(self):
        return self.rotateBoards

    # horizontal symmetric
    def getMirrorSymmetricBoard(self, board):
        reverseRows = reverseRow(board)
        return reverseRows

if __name__=="__main__":
    f = feature()
    a = np.array(range(16)).reshape([4,4])
    c = f.getMirrorSymmetricBoard(a)
    print(c)