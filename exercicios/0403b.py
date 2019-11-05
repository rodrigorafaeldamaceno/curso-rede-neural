#!/usr/bin/env python3
import numpy as np
# exercício: importar iris binário

class MLP:
  
  def __init__(self):
        # dataset
    self.ds = None
    self.d = None

    self.carregaDs()

    # hiperparâmetros
    #np.random.seed(0)
    self.eta = 0.1
    self.epocas = 5000

    sigmoide            = lambda v: 1 / (1 + np.exp(-v))
    sigmoide_           = lambda y: y*(1-y)
    binaryCrossEntropy  = lambda y, d: - d * np.log(y) - (1 - d) * np.log(1 - y)
    binaryCrossEntropy_ = lambda y, d: - (d - y) / (y*(1-y))
    leakyRelu           = np.vectorize(lambda v: v if v >= 0 else 0.01 * v)
    leakyRelu_          = np.vectorize(lambda y: 1 if y > 0 else 0.01)

    #self.f1  = leakyRelu
    #self.f1_ = leakyRelu_
    self.f1  = sigmoide
    self.f1_ = sigmoide_
    self.f2  = sigmoide
    self.f2_ = sigmoide_
    self.j   = binaryCrossEntropy
    self.j_  = binaryCrossEntropy_

    # parâmetros
    instancias = len(self.ds)
    entradas = len(self.ds[0])
    ocultos = 2 # 5
    saidas = len(self.d[0])
    self.w1 = np.random.randn(ocultos, entradas) / 2
    self.b1 = np.zeros(ocultos)
    self.w2 = np.random.randn(saidas, ocultos) / 2
    self.b2 = np.zeros(saidas)

    self.y0     = np.zeros(entradas)
    self.v1     = np.zeros(ocultos)
    self.y1     = np.zeros(ocultos)
    self.v2     = np.zeros(saidas)
    self.y2     = np.zeros(saidas)
    self.delta1 = np.zeros(ocultos)
    self.delta2 = np.zeros(saidas)


  def carregaDs(self):
      # armazenando o conteudo do arquivo na memória
      arquivo = open("datasets/iris-bin.csv", "r")
      linhas = arquivo.read().split("\n")
      linhas.pop() # última linha do dataset está em branco e deve ser removida
      arquivo.close()

      # inicializando atributos
      entradas = 4
      saidas = 1
      self.ds = np.zeros([len(linhas), entradas])
      self.d = np.zeros([len(linhas), saidas])

      # carregando atributos
      for i, linha in enumerate(linhas):
        atributos = linha.split(",")
        for j in range(0, entradas):
          self.ds[i][j] = float(atributos[j])
        for j in range(saidas):
          self.d[i][j] = float(atributos[entradas + j])

  def treina(self):
    for e in range(self.epocas):
      j = 0
      acuracia = 0
      for i in range(len(self.ds)):
        # fase forward: ativando a rede
        self.y0 = self.ds[i]
        self.v1 = np.matmul(self.w1, self.y0) + self.b1
        self.y1 = self.f1(self.v1)
        self.v2 = np.matmul(self.w2, self.y1) + self.b2
        self.y2 = self.f2(self.v2)

        # medidas de avaliação
        j += self.j(self.y2[0], self.d[i][0])/len(self.ds)
        if round(self.y2[0]) == self.d[i][0]:
          acuracia += 1/len(self.ds)

        # fase backward: camada de saída
        #self.delta2[0] = self.f2_(self.y2[0]) * self.j_(self.y2[0], self.d[i][0])
        self.delta2 = self.f2_(self.y2) * self.j_(self.y2, self.d[i])
        self.w2 = self.w2 - self.eta * np.matmul(self.coluna(self.delta2), self.linha(self.y1))
        self.b2 = self.b2 - self.eta * self.delta2

        # fase backward: camada oculta
        #self.delta1[0] = self.f1_(self.y1[0]) * (self.w2[0][0] * self.delta2[0])
        #self.delta1[1] = self.f1_(self.y1[1]) * (self.w2[0][1] * self.delta2[0])
        self.delta1 = self.f1_(self.y1) * np.matmul(self.w2.transpose(), self.delta2)
        self.w1 = self.w1 - self.eta * np.matmul(self.coluna(self.delta1), self.linha(self.y0))
        self.b1 = self.b1 - self.eta * self.delta1


      if e % 100 == 0:
        print(
          "epoca {:3d}".format(e),
          "loss {:.20f}".format(j),
          "acurácia {:.4f}".format(acuracia)
        )
        if acuracia == 1:
          break
        if np.isnan(j):
          break

  # força um vetor v a virar uma matriz linha
  def linha(self, v):
    return v.reshape(1, len(v))

  # força um vetor v a virar uma matriz coluna
  def coluna(self, v):
    return v.reshape(len(v), 1)

MLP().treina()
