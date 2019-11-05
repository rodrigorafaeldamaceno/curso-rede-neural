#!/usr/bin/env python3
import numpy as np
# exercício: treinar xor usando funções modernas
# importante: este exercício sofre de explosão do gradiente e saturação
#   o próximo exercício combate estes problemas

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
    # descomente abaixo para um exemplo que converge rapidamente
    #np.random.seed(1)

    self.relu                = lambda v: max(0, v)
    self.relu_               = lambda y: 1 if y >= 0 else 0
    self.sigmoide            = lambda v: 1 / (1 + np.exp(-v))
    self.sigmoide_           = lambda y: y*(1-y)
    self.binaryCrossEntropy  = lambda y, d: - d * np.log(y) - (1 - d) * np.log(1 - y)
    self.binaryCrossEntropy_ = lambda y, d: - (d - y) / (y * ( 1 - y))
    # nota: o denominador da derviada da entropia cruzada serve para cancelar
    # o nominador da derivada da sigmoide

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

  def treina(self):
    for e in range(self.epocas):
      j = 0
      acuracia = 0
      for i in range(len(self.ds)):
        # lendo a instância
        x1 = self.ds[i][0]
        x2 = self.ds[i][1]
        d1 = self.d[i]

        # fase forward: ativando a rede
        v1 = self.w1 * x1 + self.w3 * x2 + self.b1
        y1 = self.relu(v1)
        v2 = self.w2 * x1 + self.w4 * x2 + self.b2
        y2 = self.relu(v2)
        v3 = self.w5 * y1 + self.w6 * y2 + self.b3
        y3 = self.sigmoide(v3)

        # medidas de avaliação
        j += self.binaryCrossEntropy(y3, d1)/len(self.ds)
        if round(y3) == d1:
          acuracia += 1/len(self.ds)

        # fase backward: ajustando pesos
        # ajustando os pesos de n3 (saída)
        delta3 = self.sigmoide_(y3) * self.binaryCrossEntropy_(y3, d1)
        self.w5 = self.w5 - self.eta * y1 * delta3
        self.w6 = self.w6 - self.eta * y2 * delta3
        self.b3 = self.b3 - self.eta * delta3

        # ajustando os pesos de n1 (oculto)
        delta1 = self.relu_(y1) * (self.w5 * delta3)
        self.w1 = self.w1 - self.eta * x1 * delta1
        self.w3 = self.w3 - self.eta * x2 * delta1
        self.b1 = self.b1 - self.eta * delta1

        # ajustando os pesos de n2 (oculto)
        delta2 = self.relu_(y2) * (self.w6 * delta3)
        self.w2 = self.w2 - self.eta * x1 * delta2
        self.w4 = self.w4 - self.eta * x2 * delta2
        self.b2 = self.b2 - self.eta * delta2

      if e % 100 == 0:
        print(
          "epoca {:2d}".format(e),
          "loss {:.20f}".format(j),
          "acurácia {:.4f}".format(acuracia)
        )
        if acuracia == 1:
          break
        if np.isnan(j):
          break

Xor().treina()
