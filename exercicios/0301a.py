#!/usr/bin/env python3
import numpy as np
#import math
# exercício: implementar MLP xor usando escalares

class Xor:

  def __init__(self):
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])
    # o exemplo usará a função passo, cuja a derivada é 0.
    # redes multimacadas usam outras funções de ativação como sigmoide logística e relu
    self.passo = lambda v: 1 if v >= 0 else 0
    # self.sigmoide = lambda v: 1 / (1 + math.exp(-potencial))

  def ativa(self):
    for x in self.ds:
      v1 = x[0] + x[1] - 0.5
      y1 = self.passo(v1)
      v2 = x[0] + x[1] - 1.5
      y2 = self.passo(v2)
      v3 = y1 - y2 - 0.5
      y3 = self.passo(v3)

      print(
        "x1=", x[0],
        "x2=", x[1],
        "v1={:4.1f}".format(v1),
        "y1={:4.1f}".format(y1),
        "v2={:4.1f}".format(v2),
        "y2={:4.1f}".format(y2),
        "v3={:4.1f}".format(v3),
        "y3={:4.1f}".format(y3)
      )

Xor().ativa()
