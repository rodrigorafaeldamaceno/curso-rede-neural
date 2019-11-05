#!/usr/bin/env python3
import numpy as np
# exercício: treinar xor na forma escalar

class Xor:

  def __init__(self):
    # dataset
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])
    self.d = np.array([0., 1., 1., 0.])

    # hiperparâmetros
    self.eta = 0.1
    self.epocas = 5000
    self.sigmoide = lambda v: 1 / (1 + np.exp(-v))

    # parâmetros
    self.w1 = np.random.randn() / len(self.ds[0])
    self.w2 = np.random.randn() / len(self.ds[0])
    self.w3 = np.random.randn() / len(self.ds[0])
    self.w4 = np.random.randn() / len(self.ds[0])
    self.b1 = 0
    self.b2 = 0
    self.w5 = np.random.randn() / 2
    self.w6 = np.random.randn() / 2
    self.b3 = 0
    # pesos precisam ser inicializados aleatoriamente para quebrar simetria dos hiperplanos
    #   caso contrário, a rede não vai convergir
    # estrégias comuns de sorte: uniforme [-1, 1] ou gaussiana/normal [-3, +3]
    # inicialização xavier: sorteio gaussiano dividido pelo número de neurônios da camada anterior ao peso em questão

  def treina(self):
    for e in range(self.epocas):
      print()
      print("x1 x2     v1     y1     v2     y2     v3     y3     w1     w2     w3     w4     w5     w6     b1     b2     b3")
      for i in range(len(self.ds)):

        # lendo a instância
        x1 = self.ds[i][0]
        x2 = self.ds[i][1]
        d1 = self.d[i]

        # fase forward: ativando a rede
        v1 = self.w1 * x1 + self.w3 * x2 + self.b1
        y1 = self.sigmoide(v1)
        v2 = self.w2 * x1 + self.w4 * x2 + self.b2
        y2 = self.sigmoide(v2)
        v3 = self.w5 * y1 + self.w6 * y2 + self.b3
        y3 = self.sigmoide(v3)

        # fase backward: ajustando pesos
        # ajustando os pesos de n3 (saída)
        delta3 = y3 * (1 - y3) * (y3 - d1)
        self.w5 = self.w5 - self.eta * y1 * delta3
        self.w6 = self.w6 - self.eta * y2 * delta3
        self.b3 = self.b3 - self.eta * delta3

        # ajustando os pesos de n1 (oculto)
        delta1 = y1 * (1 - y1) * (self.w5 * delta3)
        self.w1 = self.w1 - self.eta * x1 * delta1
        self.w3 = self.w3 - self.eta * x2 * delta1
        self.b1 = self.b1 - self.eta * delta1

        # ajustando os pesos de n2 (oculto)
        delta2 = y2 * (1 - y2) * (self.w6 * delta3)
        self.w2 = self.w2 - self.eta * x1 * delta2
        self.w4 = self.w4 - self.eta * x2 * delta2
        self.b2 = self.b2 - self.eta * delta2

        print(
          "{:2d}".format(x1),
          "{:2d}".format(x2),
          "{:6.2f}".format(v1),
          "{:6.2f}".format(y1),
          "{:6.2f}".format(v2),
          "{:6.2f}".format(y2),
          "{:6.2f}".format(v3),
          "{:6.2f}".format(y3),
          "{:6.2f}".format(self.w1),
          "{:6.2f}".format(self.w2),
          "{:6.2f}".format(self.w3),
          "{:6.2f}".format(self.w4),
          "{:6.2f}".format(self.w5),
          "{:6.2f}".format(self.w6),
          "{:6.2f}".format(self.b1),
          "{:6.2f}".format(self.b2),
          "{:6.2f}".format(self.b3)
        )

Xor().treina()

