#!/usr/bin/env python3
# exercício: implementar as funções lógicas

#*******************************************************************************
# classe de base
#*******************************************************************************
class Perceptron:

  def __init__(self):
    # por conveniência, o dataset será armazenado nesta classe, em redes reais isso não ocorre
    self.ds = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]
    self.passo = lambda v: 1 if v >= 0 else 0 # representando função passo como função anônima
    #def passo(self, v):
    #  return 1 if v >= 0 else 0

  def ativaDs(self):
    for x in self.ds:
      self.ativaX(x)

  def ativaX(self, x):
    v = self.calculaV(x[0], x[1])
    y = self.passo(v)
    msg = "f(" + str(x[0]) + ", " + str(x[1]) + ") = " + str(y)
    print(msg)

  def calculaV(self, x1, x2):
    return 0 # classe filha sobrescreve


#*******************************************************************************
# e-lógico
#*******************************************************************************
class And(Perceptron):

  def calculaV(self, x1, x2):
    v = x1 + x2 - 1.5
    return v

#*******************************************************************************
# ou-lógico
#*******************************************************************************
class Or(Perceptron):

  def calculaV(self, x1, x2):
    v = x1 + x2 - 0.5
    return v

#*******************************************************************************
# implicação
#*******************************************************************************
class Implica(Perceptron):

  def calculaV(self, x1, x2):
    v = - x1 + x2 + 0.5
    return v

#*******************************************************************************
# não-lógico
# trabalha só com uma entrada, então vai sobrescrever funções da base
#*******************************************************************************
class Not(Perceptron):

  def __init__(self):
    super().__init__()    # chama construtor da classe base
    self.ds = [0, 1]


  def calculaV(self, x1):
    v = -x1 + 0.5
    return v

  def ativaX(self, x1):
    v = self.calculaV(x1)
    y = self.passo(v)
    msg = "f(" + str(x1) + ") = " + str(y)
    print(msg)

def main():
  print("rede and")
  And().ativaDs()
  print("rede or")
  Or().ativaDs()
  print("rede implica")
  Implica().ativaDs()
  print("rede not")
  Not().ativaDs()


main()
