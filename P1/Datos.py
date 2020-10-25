import pandas as pd
import numpy as np

# Excepción en caso de que algún dato no sea de tipo numérico ni nominal
class ValueError(Exception):
    def __init__(self,msg):
        self.message = msg
    def __str__(self):
        return self.message


# Función auxiliar que evalúa el tipo de un cierto array y decide si es válido o no
def isNominal(x):
    #El tipo que guarda de las Strings es Object
    if x.kind == 'O':
        return True
    # Los tipos numéricos numpy permitidos son enteros, número sin signo y float
    elif x.kind in ['i','u','f']:
        return False
    # Cualquier otro valor no está permitido
    else:
        raise ValueError(msg="Error: introducido valor distinto de entero o nominal")

# Dado un array de datos nominales, calcula un diccionario que codifica dichos valores
# en índices que ordenan dichos datos por orden alfabético
def encodeAtribute(datos):
    keys = sorted(list(set(datos)),key=str.lower)
    return {k:i for i, k in enumerate(keys)}

# Clase que guarda un conjunto de datos obtenido mediante un fichero. De esta clase se
# obtienen tres atributos que pueden usarse:
# - datos: matriz de datos guardada en formato numpy
# - nominalAtributos: array que muestra si cada columna adopta valores nominales o no (en tal caso los
#     valores son numéricos)
# - diccionario: array de diccionarios que muestra la codificación en índices de las columnas nominales
class Datos:
    # nombreFichero: dirección relativa del fichero de datos respecto a Datos.py
    # predNominal: valor que permite forzar a la última columna (la clase) a ser un
        # dato nominal aunque sea detectada como numérica. Útil si se van a realizar algoritmos
        # de clasificación
    def __init__(self, nombreFichero, predNominal=False):
        #Inicialización datos
        df = pd.read_csv(nombreFichero, header=0)
        self.datos = df.values

        #Inicialización nominalAtributos
        types = df.dtypes.array
        self.nominalAtributos = [isNominal(x) for x in types]

        # Se fuerza a las clases a ser datos nominales
        if predNominal and not self.nominalAtributos[-1]:
            self.nominalAtributos[-1] = True
            self.datos[:,-1] = np.char.mod('%d', self.datos[:,-1])

        #Inicialización diccionario
        self.diccionario = [encodeAtribute(self.datos[:,i]) if val else {} for i, val in enumerate(self.nominalAtributos)]

        # Transformación de los valores nominales en su codificación
        for i in range(np.shape(self.datos)[1]):
            if self.nominalAtributos[i]:
                self.datos[:,i] = [self.diccionario[i].get(val) for val in self.datos[:,i]]

    # Permite la obtención de ciertas filas de la matriz datos dados unos ciertos índices
    def extraeDatos(self, idx):
        return self.datos[ idx ]
