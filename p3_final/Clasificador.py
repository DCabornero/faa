from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm
from time import time
import matplotlib.pyplot as plt

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

    # Calcula las desviaciones típicas de un conjunto de datos. Para cada columna se
    # devuelve:
    # - Una dupla con la media y la desviacion típica si contiene datos continuos
    # - None si contiene datos discretos
    def calcularMediasDesv(self,datos,nominalAtributos):
        means = np.zeros(len(nominalAtributos))
        stds = np.zeros(len(nominalAtributos))
        for i in range(len(nominalAtributos)):
            if not nominalAtributos[i]:
                means[i] = np.mean(datos[:,i])
                stds[i] = np.std(datos[:,i])
        # Se podría hacer np.mean(datos, axis=0) y np.std(datos, axis=0)
        # directamente. Aunque se calcule tambien la media y desv de atributos
        # no nominales, reduce mucho el numero de lineas y es mas eficiente
        return means, stds

    # Devuelve la matriz con los datos de las columnas continuas normalizados.
    # Los datos nominales permanecen iguales.
    def normalizarDatos(self,datos,nominalAtributos):
        norm = np.copy(datos)
        for i in range(len(nominalAtributos)-1):
            if not nominalAtributos[i]:
                norm[:,i] = (norm[:,i] - self.trainMeans[i]) / self.trainStds[i]
        return norm

##############################################################################
##############################################################################

# Clase que hereda el método abstracto clasificador. Entrena y clasifica conjuntos
# de datos siguiendo la estrategia Naive-Bayes.
class ClasificadorNaiveBayes(Clasificador):

    def __str__(self):
        return 'NaiveBayes'

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
            means[i] = np.mean(arr[foo,0])
            stds[i] = np.std(arr[foo,0])
        return (means, stds)

    # Cálculo de priores dada la columna de clases del dataset.
    def procesaFinal(self, arr, numY):
        foo = np.zeros(numY)
        arrAux = arr.astype('int')
        for row in arrAux:
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


##############################################################################
##############################################################################

# Clase que hereda el método abstracto clasificador. Entrena y clasifica conjuntos
# de datos siguiendo la estrategia de regresión logística.
class ClasificadorRegresionLogistica(Clasificador):

    def __str__(self):
        return 'RegresionLogistica'

    # eta: constante de aprendizaje
    def __init__(self,eta=1,epochs=10,normaliza=True):
        self.eta = eta
        self.epochs = epochs
        self.norm = normaliza

    def sig(self,x):
        return 1/(1+np.exp(-x))

    # Entrenamiento realizado mediante la estrategia de regresión logística.
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        # Normalización de datos continuos
        # print(datostrain[:,0])
        if self.norm:
            self.trainMeans, self.trainStds = self.calcularMediasDesv(datostrain,atributosDiscretos)
            normTrain = self.normalizarDatos(datostrain,atributosDiscretos)
            # print(normTrain[:,0])
        else:
            normTrain = datostrain

        # A la matriz de entrenamiento la añadimos una columna de unos a la izquierda
        Xtrain = np.hstack((np.ones((np.shape(datostrain)[0],1)),normTrain[:,:-1]))
        ytrain = normTrain[:,-1]

        # Inicialización del vector w con valores aleatorios entre -0.5 y 0.5
        w = np.random.random_sample((np.shape(Xtrain)[1]))-0.5

        for _ in range(self.epochs):
            for i, row in enumerate(Xtrain):
                sigmoide = self.sig(np.dot(w,row))
                w = w - self.eta*(sigmoide-ytrain[i])*row

        self.trainVector = w

    # Clasificación de un cierto testSet según Regresión Logística una vez obtenido el vector
    # de entrenamiento.
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        # Normalización de datos
        if self.norm:
            normTest = self.normalizarDatos(datostest,atributosDiscretos)
        else:
            normTest = datostest

        # A la matriz de entrenamiento la añadimos una columna de unos a la izquierda
        Xtest = np.hstack((np.ones((np.shape(datostest)[0],1)),normTest[:,:-1]))

        results = np.zeros((np.shape(datostest)[0]))
        # Para cada resultado, vemos qué da el vector de entrenamiento con los parmámetros
        # de cada ejemplo. Ese resultado es la probabilidad de ser de clase 1.
        for i, row in enumerate(Xtest):
            if self.sig(np.dot(self.trainVector,row)) <= 0.5:
                results[i] = 0
            else:
                results[i] = 1
        return results

##############################################################################
##############################################################################


# Clase que hereda el método abstracto clasificador. Entrena y clasifica conjuntos
# de datos usando el algoritmo de K Vecinos próximos
class ClasificadorVecinosProximos(Clasificador):

    def __str__(self):
        return 'VecinosProximos'

    def distEuclidea(self,x, y):
        return np.sqrt(np.sum(np.power(x-y, 2)))

    def distManhattan(self,x, y):
        return np.sum(np.abs(x-y))

    def distMahalanobis(self,x, y):
        diff = x - y
        prod = np.matmul(np.matmul([diff],self.mahaMatrix),np.transpose([diff]))[0,0]
        return np.sqrt(prod)

    # numeroVecinos:
    # distancia: funcion distancia con la que se aplicará el algoritmo.
    #   Las posibles distancias son:
    #   - euclidea
    #   - mahalanobis
    #   - manhattan
    # normaliza: parámetro que indica si se normalizan o no los datos
    def __init__(self, numeroVecinos, distancia='euclidea', normaliza=True):
        # Diccionario que contiene la funciones para cada distancia
        self.distancias = {
            'euclidea': self.distEuclidea,
            'manhattan': self.distManhattan,
            'mahalanobis': self.distMahalanobis
        }
        self.numeroVecinos = numeroVecinos
        self.dist = self.distancias[distancia]
        self.norm = normaliza

    # Método de entrenamiento de k-nn. Se calcula la media y desv de los datos
    # y se guardan los datos normalizados
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        # Normalización de datos continuos
        if self.norm:
            self.trainMeans, self.trainStds = self.calcularMediasDesv(datostrain,atributosDiscretos)
            self.normTrain = self.normalizarDatos(datostrain,atributosDiscretos).astype(np.float32)
        else:
            self.normTrain = datostrain.astype(np.float32)
        # Cálculo de la inversa de la matriz de covarianzas
        if self.dist == self.distMahalanobis:
            self.mahaMatrix = np.linalg.inv(np.cov(np.transpose(self.normTrain[:,:-1])))


    # Clasifica un solo dato usando k-nn
    def clasificaUno(self,dato):
        dists = np.zeros(self.normTrain.shape[0])
        for i, row in enumerate(self.normTrain[:,:-1]):
            dists[i] = self.dist(dato, row)
        # devuelve los indices de las k menores distancias
        ind = np.argpartition(dists, self.numeroVecinos)[:self.numeroVecinos]
        # devuelve las clases de los datos que han obtenido las menores distancias
        clases = self.normTrain[ind,-1]
        # bincount devuelve las frecuencias de cada numero en un array de enteros
        # no-negativos
        valores, counts = np.unique(clases,return_counts=True)
        return valores[np.argmax(counts)]

    # Clasificación datostest usando k-nn
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        # Normalización de datos
        if self.norm:
            normTest = self.normalizarDatos(datostest,atributosDiscretos)
        else:
            normTest = datostest.astype(np.float32)
        # Array donde se van a ir guardando la clase predicha para cada dato
        preds = np.zeros((np.shape(datostest)[0]))
        for i, row in enumerate(normTest):
            preds[i] = self.clasificaUno(row[:-1])
        return preds

##############################################################################
##############################################################################

class AlgoritmoGenetico(Clasificador):
    def __str__(self):
        return 'AlgoritmoGenetico'
    # Devuelve un mismo individuo con las reglas cambiadas de sitio
    def shuffleIndividuo(self,individuo):
        numReglas = int(len(individuo)/self.lenRegla)
        ind = individuo.reshape((numReglas,self.lenRegla))
        np.random.shuffle(ind)
        return ind.reshape((len(individuo)))

    # Ordena los individuos por longitud: coloca al más corto primero
    def shortFirst(self,individuos):
        if len(individuos[0]) > len(individuos[1]):
            individuos.reverse()
        return individuos

    # Cada función de cruce devuelve una matriz con los dos individuos nuevos
    # inds: array con los dos individuos a cruzar.
    def uniforme(self,inds):
        ordered = self.shortFirst(inds)
        ordered[1] = self.shuffleIndividuo(ordered[1])

        cruceBool = np.random.randint(2,size=len(ordered[0])).astype(bool)
        cruce = np.array([i for i,res in enumerate(cruceBool) if res])

        swappedInds = [np.copy(ind) for ind in ordered]
        swappedInds.reverse()
        for i in range(2):
            ordered[i][cruce] = swappedInds[i][cruce]
        return ordered

    def unPunto(self,inds):
        ordered = self.shortFirst(inds)
        ordered[1] = self.shuffleIndividuo(ordered[1])
        punto = np.random.randint(1,len(ordered[0]))
        swapped = ordered.copy()
        swapped.reverse()
        return [np.concatenate((ordered[i][:punto],swapped[i][punto:])) for i in range(2)]

    def dosPuntos(self,inds):
        ordered = self.shortFirst(inds)
        ordered[1] = self.shuffleIndividuo(ordered[1])
        # Escogemos dos puntos DISTINTOS al azar y ORDENADOS
        punto1 = np.random.randint(1,len(ordered[0]))
        punto2 = np.random.randint(1,len(ordered[0])-1)
        if punto2 >= punto1:
            punto2 += 1
        else:
            punto1, punto2 = punto2, punto1
        swapped = ordered.copy()
        swapped.reverse()
        return [np.concatenate((ordered[i][:punto1],swapped[i][punto1:punto2],ordered[i][punto2:])) for i in range(2)]


    def __init__(self,poblacion=50,epochs=100,maxReglas=4,probMutBit=0.01,probMutInd=0.05,elitism=0.05,cruce='unPunto',plot_fitness=False):
        self.cruces = {'uniforme': self.uniforme,
                        'unPunto': self.unPunto,
                        'dosPuntos': self.dosPuntos}
        self.poblacion = poblacion
        self.epochs = epochs
        self.maxReglas = maxReglas
        self.probMutacionBit = probMutBit
        self.probMutacionInd = probMutInd
        self.elite = np.floor(poblacion*elitism).astype(int) #Se calcula directamente el número de élites por población
        self.cruce = self.cruces[cruce]
        self.plot = plot_fitness

    # Genera un individuo nuevo al azar
    def individuoRandom(self):
        longitud = self.lenRegla*np.random.randint(1,self.maxReglas)
        return np.random.randint(2,size=(longitud)).astype(bool)

    # Define una población inicial definida por una matriz de booleanos donde cada fila corresponde a un
    # individuo y cada columna a un posible valor de una regla concreta.
    def initPoblacion(self):
        numRows = self.poblacion

        randMatrix = [self.individuoRandom() for _ in range(numRows)]
        return randMatrix

    # Muta una poblacion dados sus individuos
    def mutarPoblacion(self,poblacion):
        # Matriz que determina los bits a mutar
        for i,ind in enumerate(poblacion):
            # Primero, mutación bit a bit
            randMatrix = np.random.rand(np.shape(ind)[0])
            func1 = np.vectorize(lambda x: x < self.probMutacionBit)
            mutaciones = func1(randMatrix)
            poblacion[i] = np.logical_xor(ind,mutaciones)
            # Segunda mutación: con la probabilidad de mutación, el individuo recibe
            # o elimina una regla (a cara o cruz)
            if np.random.random() < self.probMutacionInd:
                # Cara o cruz
                nReglas = int(len(ind)/self.lenRegla)
                if np.random.randint(2):
                    if nReglas < self.maxReglas:
                        newRegla = np.random.randint(2,size=(self.lenRegla)).astype(bool)
                        poblacion[i] = np.concatenate((ind,newRegla))
                else:
                    if nReglas > 1:
                        popRegla = np.random.randint(nReglas)
                        poblacion[i] = np.delete(ind,np.s_[popRegla*self.lenRegla:(popRegla+1)*self.lenRegla])
        return poblacion

    # Dado un trainSet calcula el prior, es decir, la clase más común
    def calculaPrior(self,trainSet):
        clases, count = np.unique(trainSet[:,-1],return_counts=True)
        return clases[np.argmax(count)]

    # Un individuo predice una cierta clase para un sample (sin la clase)
    # predNinguno y predEmpate son los valores que se darán si se da que
    # ninguna regla cumple los requisitos o que hay empate entre reglas (respectivamente)
    # Si se desea que siempre fallen, -1. Si se desea dar el prior, 'prior'
    def prediceClase(self,individuo,sample,predNinguno='prior',predEmpate='prior'):
        dict = {-1:None,'prior':self.prior}
        predNone = dict.get(predNinguno)
        predDraw = dict.get(predEmpate)
        # Array de matrices (numReglas x numValores), donde cada matriz corresponde a un atributo
        numReglas = int(len(individuo)/self.lenRegla)
        reglas = np.hsplit(individuo.reshape(numReglas,self.lenRegla),self.splitAtributos)
        clasesReglas = reglas[-1][:,0]
        # Array donde se guarda si un atributo del sample cumple una regla
        reglaCumplida = np.zeros((numReglas,len(sample))).astype(bool)

        for i,val in enumerate(sample):
            reglaCumplida[:,i] = reglas[i][:,val]

        # Booleanos que indican qué reglas aplican para el sample
        andSample = np.all(reglaCumplida,axis=1)

        reglasAplicadas = clasesReglas[andSample]
        values,counts = np.unique(reglasAplicadas,return_counts=True)
        # Si ninguna regla aplica se devuleve el prior
        if len(values) == 0:
            return predNone
        elif len(values) == 1:
            return values[0]
        elif counts[0] == counts[1]:
            return predDraw
        else:
            return values[np.argmax(counts)]

    # Calcula el fitness para un cierto individuo
    def fitnessInd(self,ind,trainSet,predNinguno,predEmpate):
        atributos = trainSet[:,:-1]
        clases = trainSet[:,-1]

        preds = np.apply_along_axis(lambda x: self.prediceClase(ind,x,predNinguno,predEmpate),1,atributos)
        tasaAcierto = np.mean(np.equal(preds,clases))
        return tasaAcierto

    # Calcula el fitness para una cierta población
    def fitnessPobl(self,poblacion,trainSet,predNinguno='prior',predEmpate='prior'):
        return np.array(list(map(lambda x: self.fitnessInd(x,trainSet,predNinguno,predEmpate),poblacion)))

    # Devuelve la subpoblación elitista
    def eligeElite(self,poblacion,fitness):
        eliteIndexes = np.argsort(fitness)[-self.elite:]
        return [np.copy(poblacion[i]) for i in eliteIndexes]

    # Devuelve dos individuos descendientes de acuerdo con el fitness (se pone pesos para indicar
    # que lo que se pasan son los fitness reescalados para que sumen 1)
    def eligeDescendientes(self,poblacion,pesos):
        # Selección aleatoria de los índices
        selected = np.random.choice(len(poblacion),2,replace=False,p=pesos)
        return [np.copy(poblacion[i]) for i in selected]

    def printSituacion(self,epoca,fitness):
        media = np.mean(fitness)
        maximo = np.max(fitness)
        print('====================================================')
        print('Época: ',epoca)
        print('Fitness medio: ',media)
        print('Fitness del mejor individuo: ',maximo)
        print('====================================================')
        if self.plot:
            self.fitMax[epoca] = maximo
            self.fitAvg[epoca] = media

    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        t0 = time()
        if self.plot:
            self.fitMax = np.zeros((self.epochs+1))
            self.fitAvg = np.zeros((self.epochs+1))
        # Incialización de datos iniciales: cálculo del prior, población inicial y puntos
        # de corte entre atributos dentro de las reglas
        values, count = np.unique(datosTrain[:,-1],return_counts=True)
        self.prior = values[np.argmax(count)]

        lenAtribs = np.array([len(x) for x in diccionario])
        lenAtribs[-1] = 1
        self.lenRegla = np.sum(lenAtribs)
        self.splitAtributos = np.array([np.sum(lenAtribs[:i+1]) for i in range(len(lenAtribs)-1)])

        poblacion = self.initPoblacion()

        numDescendientes = self.poblacion - self.elite
        numApareos = np.ceil(numDescendientes/2.0).astype(int)

        # Comienzan las generaciones
        for epoch in range(self.epochs):
            # Informe de situación
            fitness = self.fitnessPobl(poblacion,datosTrain,predNinguno=-1,predEmpate=-1)
            self.printSituacion(epoch,fitness)

            # Elitismo
            elite = self.eligeElite(poblacion,fitness)
            # Creación de progenitores y descendientes
            descendientes = []
            fitEscala = fitness / np.sum(fitness)
            for ap in range(numApareos):
                prog = self.eligeDescendientes(poblacion,fitEscala)
                descendientes.extend(self.cruce(prog))
            # A lo mejor hemos creado más descendientes de los necesarios, cortamos hasta donde necesitemos
            mutados = self.mutarPoblacion(descendientes[:numDescendientes])
            poblacion = elite + mutados

        fitness = self.fitnessPobl(poblacion,datosTrain,predNinguno=-1,predEmpate=-1)
        self.printSituacion(self.epochs,fitness)
        t1 = time()
        print('Tiempo de ejecución: ',t1-t0,' segundos')

        if self.plot:
            epArr = np.array([e for e in range(self.epochs+1)])
            plt.plot(epArr,self.fitMax,label='Fit. Max.')
            plt.plot(epArr,self.fitAvg,label='Fit. Avg')
            plt.legend()
            plt.show()

        # Individuo con mejor fitness
        self.individuo = poblacion[np.argmax(fitness)]

    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        func = lambda x: self.prediceClase(self.individuo,x)
        return np.apply_along_axis(func,1,datosTest[:,:-1])
