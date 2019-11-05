#!/usr/bin/env python3
# exercício: implementar somador binário
import numpy as np
import sys

class Soma:

  def __init__(self):
    # x1 é o dígito mais significativo da entrada
    x1 = int(sys.argv[1])
    x2 = int(sys.argv[2])
    x3 = int(sys.argv[3])
    x4 = int(sys.argv[4])

    # primeiro neurônio (oculto)
    v1 = x2 + x4 - 1.5
    y1 = self.passo(v1)

    # segundo neurônio (oculto)
    v2 = x1 + x3 + y1 - 1.5
    y2 = self.passo(v2)

    # terceiro neurônio (saída): o dígito mais significativo
    v3 = y2 - 0.5
    y3 = self.passo(v3)

    #  quarto neurônio (saída)
    v4 = x1 + x3 + y1 - 2*y2 - 0.5
    y4 = self.passo(v4)

    #  quinto neurônio (saída)
    v5 = x2 + x4 - 2*y1 - 0.5
    y5 = self.passo(v5)

    print(y3, y4, y5)

  def passo(self, v):
    return 1 if v >= 0 else 0


Soma()
