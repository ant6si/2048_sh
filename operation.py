import numpy as np


def getRow(board, r):
    """
    return the row
    """
    return board[r]


def getCol(board, c):

    """
    return the column
    """
    return board[:, c]


def setRow(board, r, row):
    """
    set the row, and return changed board
    """
    if r > 3:
        print("Invalid row number!")
    else:
        board[r] = row
    return board


def setCol(board, c, col):
    """
    set the column, and return changed board
    """
    if c > 3:
        print("Invalid col number!")
    else:
        board[:, c] = col
    return board


def setRows(rows):
    """
    return the board
    """
    return rows


def setCols(cols):
    """
    return the transposed board
    """
    return np.transpose(cols)


def reverseRow(row):
    """
    return mirror symmetric rows
    """
    if len(row) == 4:
        return np.flip(row, axis=0)
    else:
        return np.flip(row, axis=1)

def reverseCol(matrix):
    """
    return mirror symmetric board
    """
    return np.flip(matrix, axis=0)


def list_to_tuple(l):
    """
    change list to tuple
    """
    # only work for dimension 2
    length = len(l)
    t_list = []
    for i in range(length):
        t_list.append(tuple(l[i]))
    t = tuple(t_list)
    return t

def tuple_to_list(t):
    """
    change tuple to list
    """
    length = len(t)
    l = []
    for i in range(length):
        l.append(list(t[i]))
    return l
