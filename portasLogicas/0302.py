#!/usr/bin/env python3
# exercício: implementar as funçẽos lógicas

#*******************************************************************************
# classe de base
#*******************************************************************************
class Perceptron:

  def __init__(self):
    # por conveniência, o dataset será armazenado nesta classe, em redes reais isso não ocorre
    self.dataset = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]

  def passo(self, v):
    return 1 if v >= 0 else 0

  def f(self, x1, x2):
    return 0 # classe filha sobrescreve

  def ativaDataset(self):
    for instancia in self.dataset:
      self.ativaInstancia(instancia)

  def ativaInstancia(self, instancia):
    x1 = instancia[0]
    x2 = instancia[1]
    y = self.f(x1, x2)
    msg = "f(" + str(x1) + ", " + str(x2) + ") = " + str(y)
    print(msg)

#*******************************************************************************
# e-lógico
#*******************************************************************************
class And(Perceptron):

  def f(self, x1, x2):
    v = x1 + x2 - 1.5
    y = self.passo(v)
    return y

#*******************************************************************************
# ou-lógico
#*******************************************************************************
class Or(Perceptron):

  def f(self, x1, x2):
    v = x1 + x2 - 0.5
    y = self.passo(v)
    return y

#*******************************************************************************
# implicação
#*******************************************************************************
class Implica(Perceptron):

  def f(self, x1, x2):
    v = - x1 + x2 + 0.5
    y = self.passo(v)
    return y

#*******************************************************************************
# não-lógico
# trabalha só com uma entrada, então vai sobrescrever funções da base
#*******************************************************************************
class Not(Perceptron):

  def __init__(self):
    self.dataset = [[0], [1]]

  def f(self, x1):
    v = -x1 + 0.5
    return  self.passo(v)

  def ativaInstancia(self, instancia):
    x1 = instancia[0]
    y = self.f(x1)
    msg = "f(" + str(x1) + ") = " + str(y)
    print(msg)

def main():
  print("rede and")
  And().ativaDataset()
  print("rede or")
  Or().ativaDataset()
  print("rede implica")
  Implica().ativaDataset()
  print("rede not")
  Not().ativaDataset()


main()
