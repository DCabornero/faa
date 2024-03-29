from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm

from EstrategiaParticionado import ValidacionSimple

# Clase abstacta a partir de la cual se implementarán todos los clasificadores
class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en cada clasificador concreto
    @abstractmethod
    # La función entrenamiento permite obtener los parámetros necesarios para
    # realizar predicciones
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # Una vez entrenado, el clasificador devuleve un array de predicciones
    # dado un testSet
    # datosTest: matriz numpy con los datos de prueba
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores y devuelve la tasa de fallo
    def error(self,datos,pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        comp = np.column_stack((datos[:,-1],pred))
        failsArr = [0 if x[0] == x[1] else 1 for x in comp]
        return np.average(failsArr)


    # Realiza una clasificacion utilizando una estrategia de particionado determinada.
    # Devuelve un array con los errores de cada una de las clasificaciones hechas
    def validacion(self,particionado,dataset,clasificador=None,seed=None):
        # Se permite la creación de una clase abstracta Clasificador, de forma que mediante el parámetro clasificador
        # se introduzca la implementación de los métodos 'entrenamiento' y 'clasifica'. Se ha decidido realizar esta implementación
        # porque se incluía el parámetro clasificador en la plantilla. A priori, asumimos que estamos trabajando con una clase
        # que ya ha instanciado estas funciones.
        if clasificador == None:
            clasificador = self
        # Obtenemos las particiones según la estrategia de particionado dada
        particionado.creaParticiones(dataset,seed=seed)
        errs = np.zeros(len(particionado.particiones))
        # Para cada una de las particiones entrenamos al clasificador con el trainSet
        # y clasificamos los ejemplos del testSet a la vez que calculamos la tasa de
        # acierto de esta clasificación
        for i, p in enumerate(particionado.particiones):
            datostrain = dataset.extraeDatos( p.indicesTrain )
            datostest = dataset.extraeDatos( p.indicesTest )
            clasificador.entrenamiento( datostrain, dataset.nominalAtributos, dataset.diccionario)
            pred = clasificador.clasifica( datostest, dataset.nominalAtributos, dataset.diccionario )
            errs[i] = self.error( datostest, pred )
        return errs

    # Función que, dados una matriz de datos en formato numpy y sus predicciones
    # calcula la matriz de confusión correspondiente. Si no queda claro cual es la
    # clase positiva y cual es la negativa, la selección de falsos positivos y
    # falsos negativos es arbitraria.
    def confusion_matrix(self,datos,pred):
        conf_mat = np.zeros((2,2))
        comp = np.column_stack((datos[:,-1],pred))
        foo = {(True,True):(0,0), (True,False):(0,1), (False,True):(1,0), (False,False):(1,1)}
        for x in comp:
            conf_mat[foo[ (x[0] == 1, x[1] == 1) ]] += 1
        return conf_mat

    # Función que dada una clase Datos y la proporcion de testSet nos devuelve la
    # matriz de confusión obtenida mediante validación simple.
    def get_confusion_matrix(self,particionado,dataset,seed=None):
        # Creación de la única partición
        particionado.creaParticiones(dataset,seed=seed)
        conf_mats = np.zeros((2,2))
        for p in particionado.particiones:
            datostrain = dataset.extraeDatos( p.indicesTrain )
            datostest = dataset.extraeDatos( p.indicesTest )
            # Entrenamiento y clasificación de los datos dados
            self.entrenamiento( datostrain, dataset.nominalAtributos, dataset.diccionario )
            pred = self.clasifica( datostest, dataset.nominalAtributos, dataset.diccionario )
            conf_mats += self.confusion_matrix(datostest,pred)
        # Si utilizamos un particionado que devuelva varias particiones, se
        # calcula la matriz de confusión para cada partición y se devuelve
        # la suma de las matrices
        return conf_mats

##############################################################################


# Clase que hereda el método abstracto clasificador. Entrena y clasifica conjuntos
# de datos siguiendo la estrategia Naive-Bayes.
class ClasificadorNaiveBayes(Clasificador):

    # El parámetro opcional laplace permite que se aplique la corrección de Lapace
    # al conjunto de entrenamiento si fuera necesario.
    def __init__(self, laplace=False):
        self.laplace = laplace

    # Obtenemos a partir de una simplificación de la matriz de datos la proporcion
    # de que aparezca un cierto atributo discreto dada cada clase.
    # arr: array de duplas donde cada dupla corresponde a un ejemplo. De ese ejemplo
    # el primer dato de la dupla se corresponde con el valor que tiene un cierto atributo
    # y el segundo con la clase del ejemplo
    # numX: número de valores que adopta el atributo
    # numY: número de clases
    # laplace: booleano que indica si hay que realizar la corrección de Laplace
    # Se devuelve una matriz donde cada fila corresponde a un valor de un atributo
    # y cada columna corresponde a una clase. Como estamos mostrando la proporción
    # de datos relativos a cada clase, los valores de las columnas deben sumar 1.
    def procesaDiscreto(self, arr, numX, numY, laplace):
        foo = np.zeros((numX, numY))
        # Tabla de frecuencias
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

    # Función con las mismas características que la función anterior, pero ahora el
    # atributo toma valores continuos. La función devuelve un array de duplas de dimensión
    # el número de clases, donde cada dupla representa la media y la desviación típica
    # de los datos del atributo relativos a una cierta clase.
    def procesaContinuo(self, arr, numY):
        means = np.zeros(numY)
        stds = np.zeros(numY)
        for i in range(numY):
            foo = np.where(arr[:,1] == i)
            means[i] = np.mean(foo)
            stds[i] = np.std(foo)
        return (means, stds)

    # Cálculo de priores dada la columna de clases del dataset.
    def procesaFinal(self, arr, numY):
        foo = np.zeros(numY)
        for row in arr:
            foo[row] += 1
        return foo / np.sum(foo)

    # Entrenamiento realizado mediante la estrategia Naive-Bayes.
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        self.trainInfo = []
        # Para cada atributo, obtenemos los datos necesarios.
        for i in range(len(atributosDiscretos)-1):
            foo = np.column_stack((datostrain[:,i],datostrain[:,-1]))
            # Si el atributo es discreto, obtenemos la matriz de ocurrencias normalizada por clases.
            if atributosDiscretos[i]:
                self.trainInfo.append(self.procesaDiscreto(foo, len(diccionario[i]), len(diccionario[-1]),self.laplace))
            # Si el atributo es continuo, obtenemos la media y desviación típica del atributo
            # para cada clase.
            else:
                self.trainInfo.append(self.procesaContinuo(foo, len(diccionario[-1])))
        # Obtenemos los priores
        self.trainInfo.append(self.procesaFinal(datostrain[:,-1], len(diccionario[-1])))



    # Clasificación de un cierto testSet según Naive-Bayes una vez obtenidos los datos
    # de entrenamiento.
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        numRows = np.shape(datostest)[0]
        numCols = len(atributosDiscretos)
        clasif = np.zeros(numRows)
        # Predicción de cada dato del TestSet
        for i in range(numRows):
            preds = np.ones(len(diccionario[-1]))
            # Hallamos la probabilidad de cada atributo independiente
            for j in range(numCols-1):
                # Caso discreto
                if atributosDiscretos[j]:
                    atrMatrix = self.trainInfo[j]
                    preds = np.multiply(atrMatrix[datostest[i,j],:],preds)
                # Caso continuo
                else:
                    means, stds = self.trainInfo[j]
                    probs = [norm.pdf(datostest[i,j],loc=means[k],scale=stds[k]) for k in range(len(diccionario[-1]))]
                    preds = np.multiply(probs, preds)
            # Hallamos la probabilidad del prior
            preds = np.multiply(self.trainInfo[-1],preds)
            # Elección de la clase con mejor probabilidad a posteriori
            clasif[i] = np.argmax(preds)
        return clasif
