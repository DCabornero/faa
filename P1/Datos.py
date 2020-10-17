import pandas as pd
import numpy as np

class ValueError(Exception):
  def __init__(self,msg):
      self.message = msg
  def __str__(self):
      return self.message



def isNominal(x):
  #El tipo que guarda de las Strings es Object
  if x.kind == 'O':
    return True
  elif x.kind in ['i','u','f']:
    return False
  else:
    raise ValueError(msg="Error: introducido valor distinto de entero o nominal")

def encodeAtribute(datos):
  keys = sorted(list(set(datos)),key=str.lower)
  return {k:i for i, k in enumerate(keys)}


class Datos:

  def __init__(self, nombreFichero):
    #Inicialización datos
    df = pd.read_csv(nombreFichero, header=0)
    self.datos = df.values

    #Inicialización nominalAtributos
    types = df.dtypes.array
    self.nominalAtributos = [isNominal(x) for x in types]

    #Inicialización diccionario
    self.diccionario = [encodeAtribute(self.datos[:,i]) if val else {} for i, val in enumerate(self.nominalAtributos)]

    for i in range(np.shape(self.datos)[1]):
        if self.nominalAtributos[i]:
            self.datos[:,i] = [self.diccionario[i].get(val) for val in self.datos[:,i]]

  def extraeDatos(self, idx):
    pass
