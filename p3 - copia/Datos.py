import pandas as pd
import numpy as np

# Excepción en caso de que algún dato no sea de tipo numérico ni nominal
class InvalidTypeError(Exception):
    def __init__(self,msg):
        super().__init__(msg)


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
        raise InvalidTypeError("Error: introducido valor distinto de entero o nominal")

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
    # allNominal: valor que fuerza a todas las clases a ser nominales
    def __init__(self, nombreFichero, predNominal=False, allNominal=False):
        #Inicialización datos
        df = pd.read_csv(nombreFichero, header=0)
        self.datos = np.zeros(df.shape)

        if allNominal:
            df = df.astype(str)

        #Inicialización nominalAtributos
        types = df.dtypes.array
        self.nominalAtributos = [isNominal(x) for x in types]

        # Se fuerza a las clases a ser datos nominales
        if predNominal and not self.nominalAtributos[-1]:
            self.nominalAtributos[-1] = True
            df.iloc[:,-1] = df.iloc[:,-1].astype(str)

        #Inicialización diccionario
        self.diccionario = [encodeAtribute(df.iloc[:,i].values) if val else {} for i, val in enumerate(self.nominalAtributos)]

        # Transformación de los valores nominales en su codificación
        for i in range(np.shape(self.datos)[1]):
            if self.nominalAtributos[i]:
                self.datos[:,i] = [self.diccionario[i].get(val) for val in df.iloc[:,i]]
            else:
                self.datos[:,i] = df.iloc[:,i].values

        # Si elegimos allNominal, podemos interpretar el numpy array como array de enteros
        self.datos = self.datos.astype(int)

    # Permite la obtención de ciertas filas de la matriz datos dados unos ciertos índices
    def extraeDatos(self, idx):
        return self.datos[ idx ]
