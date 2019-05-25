import numpy as np
#
# class operation():
#     """
#     :param
#     board: 4X4 matrix
#     r: number of row
#     c: number of col
#     row: 1X4 vector
#     col: 1X4 vector
#     """
#     def getRow(self, board, r):
#         return board[r]
#
#     def getCol(self, board, c):
#         return board[:,c]
#
#     def setRow(self, board, r, row):
#         if r>3:
#             print("Invalid row number!")
#         else:
#             board[r] = row
#         return board
#
#     def setCol(self, board, c, col):
#         if c>3:
#             print("Invalid col number!")
#         else:
#             board[:,c] = col
#         return board
#
#     def setRows(self, rows):
#         return rows
#
#     def setCols(self, cols):
#         return cols
#
#     def reverseRow(self, row):
#         return np.flip(row)
#
# if __name__ == "__main__":
#     op = operation()
#     board = np.array(range(16)).reshape([4,4])
#     row = np.zeros(4, dtype=int)
#     col = np.ones(4, dtype=int)

def getRow(board, r):
    return board[r]

def getCol(board, c):
    return board[:, c]


def setRow(board, r, row):
    if r > 3:
        print("Invalid row number!")
    else:
        board[r] = row
    return board


def setCol(board, c, col):
    if c > 3:
        print("Invalid col number!")
    else:
        board[:, c] = col
    return board


def setRows(rows):
    return rows


def setCols(cols):
    return np.transpose(cols)


def reverseRow(row):
    if len(row) == 4:
        return np.flip(row)
    else:
        return np.flip(row, axis=1)

def reverseCol(matrix):
    return np.flip(matrix, axis=0)


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
