#!/usr/bin/env python3
import numpy as np
#import math
# exercício: implementar MLP xor usando tensores

class Xor:

  def __init__(self):
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])

    w1 = np.array([[1., 1.], [1., 1.]]) # pesos da primeira camada
    b1 = np.array([1., -1.])            # biases da primeira camada
    w2 = np.array([1., -1.])            # pesos da segunda camada
    b2 = np.array([ -0.5 ])             # biases da segunda camada

    self.w = [w1, w2]
    self.b = [b1, b2]
    self.passo = np.vectorize(lambda potencial: 1 if potencial >= 0 else 0)
    # self.sigmoide = lambda v: 1 / (1 + math.exp(-potencial))

  def ativa(self):
    for x in self.ds:
      # ativando 1a camada
      v1 = np.matmul(self.w[0], x) + self.b[0]
      y1 = self.passo(v1)
      # ativando 2a camada
      v2 = np.matmul(self.w[1], y1) + self.b[1]
      y2 = self.passo(v2)

      print(
        "x={:5s}".format(str(x)),
        "v1={:9s}".format(str(v1)),
        "y1={:7s}".format(str(y1)),
        "v2={:6s}".format(str(v2)),
        "y={:7s}".format(str(y2))
      )

Xor().ativa()
