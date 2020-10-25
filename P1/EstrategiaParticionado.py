from abc import ABCMeta,abstractmethod
import numpy as np
import random

# Clase que permite separar índices de un conjunto de datos en dos arrays: el de entrenamiento
# y el de test.
class Particion():

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

#####################################################################################################

# Clase abstacta diseñada para elaborar a partir de ella distintas estrategias que
# permiten crear una o varias particiones de datos
class EstrategiaParticionado:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # Función abstacta encargada de crear las particiones que se quieran. Devuelve una clase Particion
    # datos: clase Datos que contiene los datos que se quieren particionar
    # seed: semilla que permite controlar los barajeos de ejemplos
    def creaParticiones(self,datos,seed=None):
        pass


#####################################################################################################
# Estrategia de particionado que permite aplicar el método de validación simple.
class ValidacionSimple(EstrategiaParticionado):

    # proporcionTest: proporcion de datos que se quieren destinar al testSet frente al trainSet
    # numeroEjecuciones: número de veces que se quiere particionar la base de datos
    def __init__(self, proporcionTest, numeroEjecuciones=1):
        # TODO: comprobar que esta entre 0 y 1
        self.particiones = []
        self.proporcionTest = proporcionTest
        self.numeroEjecuciones = numeroEjecuciones

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Añade el atributo 'particiones' a la clase. 'particiones' es un array de clases Particion.
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        numDatos = len(datos.datos)
        indexList = [i for i in range(numDatos)]
        numTest = int(np.floor(numDatos * self.proporcionTest))
        self.particiones = []
        # Cálculo de las 'numeroEjecuciones' particiones que se quieran crear.
        for _ in range(self.numeroEjecuciones):
            p = Particion()
            random.shuffle(indexList)
            p.indicesTest = indexList[:numTest]
            p.indicesTrain = indexList[numTest:]
            self.particiones.append(p)



#####################################################################################################
# Estrategia de particionado que permite aplicar el método de validación cruzada.
class ValidacionCruzada(EstrategiaParticionado):
    # numeroParticiones: cantidad de particiones que se hacen a los datos.
    def __init__(self, numeroParticiones):
        self.particiones = []
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el método de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion crea el atributo de la clae particiones, una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        numDatos = len(datos.datos)
        indexList = [i for i in range(numDatos)]
        random.shuffle(indexList)
        # Cálculo del tamaño de cada bloque de datos y separación en ese tamaño
        chunkSize = int(np.floor(numDatos / self.numeroParticiones))
        bloques = [ indexList[i*chunkSize:(i + 1)*chunkSize] for i in range(self.numeroParticiones - 1) ]
        bloques.append( indexList[(self.numeroParticiones - 1)*chunkSize:] )
        self.particiones = []
        # Creación de las particiones. En cada una se utiliza un bloque como testSet y el resto como trainSet.
        # Por ello, el número de particiones es el número de bloques.
        for i in range(len(bloques)):
            p = Particion()
            p.indicesTest = bloques[i]
            p.indicesTrain = sum([bloques[j] for j in range(len(bloques)) if j != i],[])
            self.particiones.append(p)


if __name__ == "__main__":
    validacion = ValidacionCruzada(5)
    validacion.creaParticiones([i for i in range(25)])
    for i in range(len(validacion.particiones)):
        print(validacion.particiones[i].indicesTrain)
        print(validacion.particiones[i].indicesTest)
