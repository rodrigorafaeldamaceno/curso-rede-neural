#!/usr/bin/env python3
import numpy as np
#import math
# exercício: implementar MLP xor em um único tensor

class Xor:

  def __init__(self):
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])

    w1 = np.array([[1., 1.], [1., 1.]]) # pesos da primeira camada
    b1 = np.array([-0.5, -1.5])         # biases da primeira camada
    c1 = [w1, b1]                       # representa a primeira camada
    w2 = np.array([1., -1.])            # pesos da segunda camada
    b2 = np.array([ -0.5 ])             # biases da segunda camada
    c2 = [w2, b2]                       # representa a segunda camada
    self.c = [c1, c2]                   # representa a rede de duas camadas

    self.passo = np.vectorize(lambda potencial: 1 if potencial >= 0 else 0)
    # self.sigmoide = lambda v: 1 / (1 + math.exp(-potencial))

  def ativa(self):
    # obs: o código abaixo é genérico, independe do número de camadas da rede
    for x in self.ds:
      sinal = x
      for c in self.c:
        entrada = sinal                 # útil para dar o  print
        w = c[0]
        b = c[1]
        v = np.matmul(w, sinal) + b
        y = self.passo(v)
        sinal = y
        print(
          "x={:6s}".format(str(x)),
          "sinal={:6s}".format(str(entrada)),
          "w={:25s}".format(str(w.tolist())),
          "b={:12s}".format(str(b)),
          "v={:12s}".format(str(v)),
          "y={:5s}".format(str(y))
        )
      print()
Xor().ativa()

# aprofundamento
# o código acima corresponde ao seguinte tensor de ordem 3:
#self.c = [
#  [                                 # primeira camada
#    np.array([[1., 1.], [1., 1.]]), # pesos
#    np.array([ -0.5, -1.5 ])        # biases
#  ],
#  [                                 # segunda camada
#    np.array([1., -1.]),            # pesos
#    np.array([ -0.5 ])              # biases
#  ],
#]
