import numpy as np


def passo(v):
    if(v >= 0):
        return 1
    return 0


X = np.array([[1.0, 0.0]]).transpose()

w = np.array([[1.0, 1.0],
              [1.0, 1.0]])

b = np.array([[-1.5, -0.5]]).transpose()

v = np.matmul(w, X) + b

for resultado in v:
    print(passo(resultado))
