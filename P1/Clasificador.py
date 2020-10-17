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

    def procesaDiscreto(arr, numX, numY):
        mat = np.zeros((numX, numY))
        for el in arr:
            mat[el[0], el[1]] += 1
        return mat

    def procesaContinuo(arr, numY):
        means = np.zeros(numY)
        vars = np.zeros(numY)
        for i in range(numY):
            foo = np.where(arr[:,1] == i)
            means[i] = np.mean(foo)
            vars[i] = np.var(foo)
        return (means, vars)

    # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        self.entrenamientoAux = []
        for i in range(len(atributosDiscretos)-1):
            foo = np.column_stack((datostrain[:,i],datostrain[:,-1]))
            if atributosDiscretos[i]:
                self.entrenamientoAux.append(procesaDiscreto(foo, len(diccionario[i].keys), len(diccionario[-1].keys)))
            else:
                self.entrenamientoAux.append(procesaContinuo(foo), len(diccionario[-1].keys))



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        pass
