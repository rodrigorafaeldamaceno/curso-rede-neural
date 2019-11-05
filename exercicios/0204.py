#!/usr/bin/env python3
# exercício: treinar um neurônio perceptron
import numpy as np
import math

class Perceptron:

  def __init__(self):
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])

    # parâmetros
    self.d = np.array([0., 0., 0., 1.])
    self.w = np.array([0., 0.])
    self.b = 0.
    self.passo = lambda potencial: 1 if potencial >= 0 else 0

    # hiperparâmetros
    self.eta = 0.3 # taxa de aprendizado
    self.epocas = 10

  def treina(self):
    for e in range(self.epocas):
      #print("época " + str(e) + ":")
      for i in range(len(self.ds)):
        # ativando instância
        x = self.ds[i]
        d = self.d[i]
        v = np.matmul(self.w, x) + self.b
        y = self.passo(v)

        # ajustando pesos
        self.w[0] = self.w[0] - self.eta * x[0] * (y - d)
        self.w[1] = self.w[1] - self.eta * x[1] * (y - d)
        self.b = self.b - self.eta * (y - d)
        erro = y - d

        # exibindo resultados
        print(
          "época " + str(e) + ":" +
          " x1=" + str(x[0]) +
          " x2=" + str(x[1]) +
          " d="  + str(d) +
          " y="  + str(y) +
          " e={:5.2f}".format(erro) +
          " w1={:5.2f}".format(self.w[0]) +
          " w2={:5.2f}".format(self.w[1]) +
          " b={:5.2f}".format(self.b)
        )

Perceptron().treina()
