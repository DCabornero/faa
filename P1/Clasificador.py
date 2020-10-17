from abc import ABCMeta,abstractmethod
import numpy as np


class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self,datos,pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        comp = np.column_stack((datos[:,-1],pred))
        failsArr = [0 if x[0] == x[1] else 1 for x in comp]
        return np.average(failsArr)


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self,particionado,dataset,clasificador,seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        pass

##############################################################################



class ClasificadorNaiveBayes(Clasificador):

    def procesaDiscreto(self, arr, numX, numY):
        foo = np.zeros((numX, numY))
        for row in arr:
            foo[row[0], row[1]] += 1
        return foo / np.sum(foo)

    def procesaContinuo(self, arr, numY):
        means = np.zeros(numY)
        vars = np.zeros(numY)
        for i in range(numY):
            foo = np.where(arr[:,1] == i)
            means[i] = np.mean(foo)
            vars[i] = np.var(foo)
        return (means, vars)

    def procesaFinal(self, arr, numY):
        foo = np.zeros(numY)
        for row in arr:
            foo[row] += 1
        return foo / np.sum(foo)

    # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        self.entrenamientoAux = []
        for i in range(len(atributosDiscretos)-1):
            foo = np.column_stack((datostrain[:,i],datostrain[:,-1]))
            if atributosDiscretos[i]:
                self.entrenamientoAux.append(self.procesaDiscreto(foo, len(diccionario[i]), len(diccionario[-1])))
            else:
                self.entrenamientoAux.append(self.procesaContinuo(foo), len(diccionario[-1]))
        self.entrenamientoAux.append(self.procesaFinal(datostrain[:,-1], len(diccionario[-1])))



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        pass
