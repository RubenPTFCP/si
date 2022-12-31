import numpy as np


def rmse(y_true:np.array, y_pred:np.array) -> float:
    return np.sqrt(np.sum(np.subtract(y_true, y_pred) ** 2) / len(y_true))

m1 = np.array([[3, 0, 9],
               [1, 1, 1],
               [2, 3, 4]])

m2 = np.array([[6, 6, 6],
               [9, 9, 9],
               [7, 6, 5]])


teste = rmse(m1,m2)
print(teste)