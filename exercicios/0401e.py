#!/usr/bin/env python3
import numpy as np
# exercício: combatendo explosão do gradiente e instabilidade numérica

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
    np.random.seed(10)

    # relu satura e trava em pequenos datasets quando pesos ficam muito negativos
    #self.relu                = lambda v: max(0, v)
    #self.relu)               = lambda y: 1 if y > 0 else 0

    # leaky relu é uma alternativa melhor
    self.leakyRelu           = lambda v: v if v >= 0 else 0.01 * v
    self.leakyRelu_          = lambda y: 1 if y > 0 else 0.01
    self.sigmoide            = lambda v: 1 / (1 + np.exp(-v))
    self.sigmoide_           = lambda y: y*(1-y)
    # versão numericamente estável da sigmoide logística (menos sucestível a problemas com gradiente)
    self.sigmoide           = lambda v: (1. / (1. + np.exp(-v))) if v >= 0 else (np.exp(v) / (1. + np.exp(v)))

    self.binaryCrossEntropy  = lambda y, d: - d * np.log(y) - (1 - d) * np.log(1 - y)
    self.binaryCrossEntropy_ = lambda y, d: - (d - y) / (y*(1-y))

    # parâmetros
    self.w1 = np.random.randn() / len(self.ds[0])
    self.w2 = np.random.randn() / len(self.ds[0])
    self.w3 = np.random.randn() / len(self.ds[0])
    self.w4 = np.random.randn() / len(self.ds[0])
    self.b1 = 0
    self.b2 = 0
    self.w5 = np.random.randn() / len(self.ds[0])
    self.w6 = np.random.randn() / len(self.ds[0])
    self.b3 = 0

  def treina(self):
    y = np.zeros(4)
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
        y1 = self.leakyRelu(v1)
        v2 = self.w2 * x1 + self.w4 * x2 + self.b2
        y2 = self.leakyRelu(v2)
        v3 = self.w5 * y1 + self.w6 * y2 + self.b3
        y3 = self.sigmoide(v3)

        # medidas de avaliação
        j += self.binaryCrossEntropy(y3, d1)/len(self.ds)
        if round(y3) == d1:
          acuracia += 1/len(self.ds)
        y[i] = y3

        # fase backward: ajustando pesos
        # ajustando os pesos de n3 (saída)
        delta3 = self.sigmoide_(y3) * self.binaryCrossEntropy_(y3, d1)
        self.w5 = self.w5 - self.eta * y1 * delta3
        self.w6 = self.w6 - self.eta * y2 * delta3
        self.b3 = self.b3 - self.eta * delta3

        # ajustando os pesos de n1 (oculto)
        delta1 = self.leakyRelu_(y1) * (self.w5 * delta3)
        self.w1 = self.w1 - self.eta * x1 * delta1
        self.w3 = self.w3 - self.eta * x2 * delta1
        self.b1 = self.b1 - self.eta * delta1

        # ajustando os pesos de n2 (oculto)
        delta2 = self.leakyRelu_(y2) * (self.w6 * delta3)
        self.w2 = self.w2 - self.eta * x1 * delta2
        self.w4 = self.w4 - self.eta * x2 * delta2
        self.b2 = self.b2 - self.eta * delta2

      if e % 100 == 0:
        print(
          "epoca {:3d}".format(e),
          "loss {:.20f}".format(j),
          "acurácia {:.4f}".format(acuracia),
          "{:.4f}".format(y[0]),
          "{:.4f}".format(y[1]),
          "{:.4f}".format(y[2]),
          "{:.4f}".format(y[3])
        )
        if acuracia == 1:
          break
        if np.isnan(j):
          break

Xor().treina()

"""
aprofundamento: porque essa rede converge pouco?

1 é necessário calibrar muito bem a taxa de aprendizado e o número de épocas
2 uma topologia com mais neurônios ocultos permitiria que a rede convirga mais rapidamento
  porém é mais sucestível a overfitting também
3 o exemplo usa o modo online, quando os modos batch ou offiline são mais eficientes
4 usa-se a versão mais tradicional de algoritmo otimizador (gradiente descentende)
  porém existem modificações dele como adam e rmsprop
5 este exercício é particularmente sucestível a explosão do gradiente
  técnicas mais avançadas ainda não vistas como gradient crop ou regularização l2
  podem ser usadas para resolver o problema
"""

"""
dica para ajudar no próximo exercício
self.d          -> transformar em vetor coluna
self.relu       -> self.f1
self.relu_      -> self.f1_
self.sigmoide   -> self.f2
self.sigmoide_  -> self.f2_
self.w1         -> self.w1[0][0]
self.w2         -> self.w1[1][0]
self.w3         -> self.w1[0][1]
self.w4         -> self.w1[1][1]
self.w5         -> self.w2[0][0]
self.w6         -> self.w2[0][1]
self.b1         -> self.b1[0]
self.b2         -> self.b1[1]
self.b3         -> self.b2[0]
v1              -> self.v1[0]
v2              -> self.v1[1]
v3              -> self.v2[0]
x1              -> self.y0[0]
x2              -> self.y0[1]
y1              -> self.y1[0]
y2              -> self.y1[1]
y3              -> self.y2[0]
delta1          -> self.delta1[0]
delta2          -> self.delta1[1]
delta3          -> self.delta2[0]
"""
