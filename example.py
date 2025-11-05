import numpy as np

board = np.zeros((3, 3, 3, 3), dtype=int)

X_large = 1
Y_large = 1

x = 2
y = 2

board[X_large, Y_large, x, y] = 1 # Krzyżyk

X_large = x
Y_large = y




