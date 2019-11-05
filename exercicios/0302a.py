#!/usr/bin/env python3
# exercício: implementar testador de paridade impar (n=4) usando escalares
import numpy as np
import sys

class ParidadeImpar:

  def __init__(self):
    self.passo = lambda v: 1 if v >= 0 else 0
    self.x1 = int(sys.argv[1])
    self.x2 = int(sys.argv[2])
    self.x3 = int(sys.argv[3])
    self.x4 = int(sys.argv[4])

  def ativa(self):
    v1 = self.x1 + self.x2 + self.x3 + self.x4 - 0.5
    y1 = self.passo(v1)
    v2 = self.x1 + self.x2 + self.x3 + self.x4 - 1.5
    y2 = self.passo(v2)
    v3 = self.x1 + self.x2 + self.x3 + self.x4 - 2.5
    y3 = self.passo(v3)
    v4 = self.x1 + self.x2 + self.x3 + self.x4 - 3.5
    y4 = self.passo(v4)
    v5 = y1 - y2 + y3 - y4 - 0.5
    y5 = self.passo(v5)

    print(
      "x1=", self.x1,
      " x2=", self.x2,
      " x3=", self.x3,
      " x4=", self.x4,
      " y1=", y1,
      " y2=", y2,
      " y3=", y3,
      " y4=", y4,
      " y5=", y5, sep=""
    )

ParidadeImpar().ativa()
