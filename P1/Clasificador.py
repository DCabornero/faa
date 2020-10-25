from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm

from EstrategiaParticionado import ValidacionSimple

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
    def validacion(self,particionado,dataset,clasificador=None,seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        if clasificador == None:
            clasificador = self
        particionado.creaParticiones(dataset,seed=seed)
        errs = np.zeros(len(particionado.particiones))
        for i, p in enumerate(particionado.particiones):
            datostrain = dataset.extraeDatos( p.indicesTrain )
            datostest = dataset.extraeDatos( p.indicesTest )
            clasificador.entrenamiento( datostrain, dataset.nominalAtributos, dataset.diccionario)
            pred = clasificador.clasifica( datostest, dataset.nominalAtributos, dataset.diccionario )
            errs[i] = self.error( datostest, pred )
        return errs

    def confusion_matrix(self,datos,pred):
        conf_mat = np.zeros((2,2))
        comp = np.column_stack((datos[:,-1],pred))
        foo = {(True,True):(0,0), (True,False):(0,1), (False,True):(1,0), (False,False):(1,1)}
        for x in comp:
            conf_mat[foo[(x[0] == 1, x[1] == 1)]] += 1
        return conf_mat

    def get_confusion_matrix(self,dataset,seed=None,proporcionTest=0.3):
        particionado = ValidacionSimple(proporcionTest)
        particionado.creaParticiones(dataset,seed=seed)
        p = particionado.particiones[0]
        datostrain = dataset.extraeDatos( p.indicesTrain )
        datostest = dataset.extraeDatos( p.indicesTest )
        self.entrenamiento( datostrain, dataset.nominalAtributos, dataset.diccionario )
        pred = self.clasifica( datostest, dataset.nominalAtributos, dataset.diccionario )
        return self.confusion_matrix( datostest, pred )

##############################################################################



class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        self.laplace = laplace

    def procesaDiscreto(self, arr, numX, numY, laplace):
        foo = np.zeros((numX, numY))
        for row in arr:
            foo[row[0], row[1]] += 1
        #Corrección de Laplace
        if laplace:
            numCeros = np.sum(foo == 0)
            if numCeros > 0:
                foo += 1
        #Normalizamos cada columna
        for i in range(np.shape(foo)[1]):
            foo[:,i] /= np.sum(foo[:,i])
        return foo

    def procesaContinuo(self, arr, numY):
        means = np.zeros(numY)
        stds = np.zeros(numY)
        for i in range(numY):
            foo = np.where(arr[:,1] == i)
            means[i] = np.mean(foo)
            stds[i] = np.std(foo)
        return (means, stds)

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
                self.entrenamientoAux.append(self.procesaDiscreto(foo, len(diccionario[i]), len(diccionario[-1]),self.laplace))
            else:
                self.entrenamientoAux.append(self.procesaContinuo(foo, len(diccionario[-1])))
        self.entrenamientoAux.append(self.procesaFinal(datostrain[:,-1], len(diccionario[-1])))



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        numRows = np.shape(datostest)[0]
        numCols = len(atributosDiscretos)
        clasif = np.zeros(numRows)
        # Predicción de cada dato del TestSet
        for i in range(numRows):
            preds = np.ones(len(diccionario[-1]))
            # Hallamos la probabilidad de cada atributo independiente
            for j in range(numCols-1):
                if atributosDiscretos[j]:
                    atrMatrix = self.entrenamientoAux[j]
                    preds = np.multiply(atrMatrix[datostest[i,j],:],preds)
                else:
                    probs = []
                    means, stds = self.entrenamientoAux[j]
                    probs = [norm.pdf(datostest[i,j],loc=means[k],scale=stds[k]) for k in range(len(diccionario[-1]))]
                    preds = np.multiply(probs, preds)
            # Hallamos la probabilidad del prior
            preds = np.multiply(self.entrenamientoAux[-1],preds)
            clasif[i] = np.argmax(preds)
        return clasif
