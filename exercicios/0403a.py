#!/usr/bin/env python3
import numpy as np
# exercício: importar iris binário

class MLP:

  def __init__(self):
    self.ds = None # inicialização de atributos opcional neste ponto
    self.d = None
    self.carregaDs()
    for i in range(len(self.ds)):
      print(self.ds[i], self.d[i])

  def carregaDs(self):
    # armazenando o conteudo do arquivo na memória
    arquivo = open("iris-bin.csv", "r")
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

MLP()
