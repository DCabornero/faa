from abc import ABCMeta,abstractmethod
import numpy as np
import random

class Particion():

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self,datos,seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self, proporcionTest, numeroEjecuciones=1):
        # TODO: comprobar que esta entre 0 y 1
        self.particiones = []
        self.proporcionTest = proporcionTest
        self.numeroEjecuciones = numeroEjecuciones

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        numDatos = len(datos)
        indexList = [i for i in range(numDatos)]
        numTest = int(np.floor(numDatos * self.proporcionTest))
        for _ in range(self.numeroEjecuciones):
            p = Particion()
            random.shuffle(indexList)
            p.indicesTest = indexList[:numTest]
            p.indicesTrain = indexList[numTest:]
            self.particiones.append(p)



#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, numeroParticiones):
        self.particiones = []
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        numDatos = len(datos)
        indexList = [i for i in range(numDatos)]
        random.shuffle(indexList)
        chunkSize = int(np.floor(numDatos / self.numeroParticiones))
        bloques = [ indexList[i*chunkSize:(i + 1)*chunkSize] for i in range(self.numeroParticiones - 1) ]
        bloques.append( indexList[(self.numeroParticiones - 1)*chunkSize:] )
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
