#!/usr/bin/env python3
# exercício: usar matrizes para implementar rna com dois neurônios (and e or)
import numpy as np

class RNA:

  def __init__(self):
    self.w = np.array([             # arrays numpy são mais eficitentes
      [1., 1.],                     # criando array de float no numpy
      [1., 1.]
    ])
    self.b = np.array([-1.5, -0.5]) # convenção: bias é vetor coluna (aprofundamento abaixo)
    self.dataset = np.array([       # conveniência: dataset armazenado na classe RNA (em redes reais isso não ocorre)
      [0, 0],                       # convenção: instância é vetor coluna (aprofundamento abaixo)
      [0, 1],
      [1, 0],
      [1, 1]
    ])
    # alternativa de 1 linha que faz o mesmo que a função passo mais abaixo
    # self.passo = np.vectorize(lambda v: 1 if v >= 0 else 0)

  def ativaDataset(self):
    for x in self.dataset:
      self.ativaInstancia(x)

  def ativaInstancia(self, x):
    y = self.f(x)
    msg = "f(" + str(x) + ") = " + str(y)
    print(msg)

  def f(self, x):
    v = np.matmul(self.w, x) + self.b
    # print("v = " + str(v)) # descomente para ver os potenciais de ativação
    return self.passo(v)

  def passo(self, v):
    y = np.zeros(v.shape) # y tem as mesmas dimensões de v e é preenchido com zeros
    for i in range(len(v)):
      if v[i] >= 0:
        y[i] = 1
    return y

RNA().ativaDataset()

# aprofundamento
# representações numpy
# x = np.array([x1, x2])      -> vetor linha ou coluna (conforme o contexto)
# x = np.array([[x1, x2]])    -> forçando um vetor linha
# x = np.array([[x1], [x2]])  -> forçando um vetor coluna

# multiplicação matrizes e vetores:
# x =  np.array([x1, x2])
# w =  np.array([[w11, w12],[w21, w22]])
# v =  np.matmul(w, x)
# v == np.array([w11x1 + w12x2, w21x1 + w22x2])
# observe que cada linha i de w corresponde aos pesos de entrada do neurônio i