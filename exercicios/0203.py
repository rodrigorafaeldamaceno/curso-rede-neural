#!/usr/bin/env python3
# exercício: usar matrizes para implementar rna com dois neurônios (and e or)
# obs: para manter a simplicidade, este exercício usa poucos métodos
import numpy as np

class RNA:

  def __init__(self):
    # arrays numpy são mais eficitentes
    # criando array de floats (1.)
    self.w = np.array([
      [1., 1.],
      [1., 1.]
    ])
    # convenção: bias é vetor coluna
    self.b = np.array([-1.5, -0.5])
    # convenção: instância é vetor coluna
    self.ds = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ])
    # np.vetorize permite passar arrays numpy para funções python
    self.passo = np.vectorize(lambda v: 1 if v >= 0 else 0)

  def ativa(self):
    for x in self.ds:
      # ativando neurônio
      v = np.matmul(self.w, x) + self.b
      y = self.passo(v)

      # printando resultado
      msg = "f(" + str(x) + ") = " + str(y)
      print(msg)

RNA().ativa()

# aprofundamento vetores linha e coluna
# representações numpy
# x = np.array([x1, x2])      -> vetor linha ou coluna (conforme o contexto)
# x = np.array([[x1, x2]])    -> forçando um vetor linha
# x = np.array([[x1], [x2]])  -> forçando um vetor coluna

# multiplicação matrizes e vetores:
# x =  np.array([x1, x2])
# w =  np.array([[w11, w12],[w21, w22]])
# v =  np.matmul(w, x)
# v == np.array([w11x1 + w12x2, w21x1 + w22x2])
# observe que cada linha i de w corresponde aos pesos de entrada do neurônio i (instância é vetor coluna)
