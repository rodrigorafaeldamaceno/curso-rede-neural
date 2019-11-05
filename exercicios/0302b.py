#!/usr/bin/env python3
# exercício: implementar testador de paridade impar (n=4) usando tensores
import numpy as np
import sys

class ParidadeImpar:

  def __init__(self):
    self.x = np.array([
      int(sys.argv[1]),
      int(sys.argv[2]),
      int(sys.argv[3]),
      int(sys.argv[4])
    ])
    self.w = [
      np.ones([4, 4], dtype=float),       # forma prática de inicilizar a 1a camada
      np.array([1., -1., 1., -1.])        # inicialização da 2a camada
    ]
    self.b = [
      np.array([-0.5, -1.5, -2.5, -3.5]), # 1a camada
      np.array([-0.5])                    # 2a camada
    ]
    self.passo = np.vectorize(lambda potencial: 1 if potencial >= 0 else 0)

  def ativa(self):
    # potenciais e ativações da camada oculta
    sinal = self.x
    for i in range(len(self.w)):
      print(sinal, end=" ")
      v = np.matmul(self.w[i], sinal) + self.b[i]
      y = self.passo(v)
      sinal = y
    print(sinal)

ParidadeImpar().ativa()
