import numpy as np
import matplotlib.pyplot as plt

def acuracia(coef,Xteste,Yteste):
    Ypred = np.argmax(Xteste @ coef,axis=1)
  
    acuracia = (np.mean(Ypred == np.argmax(Yteste,axis=1)))

    return acuracia

def Ols(Xtreino,Ytreino,Xteste,Yteste):
    coef = np.linalg.pinv(Xtreino.T @ Xtreino) @ Xtreino.T @ Ytreino

    ac = acuracia(coef,Xteste,Yteste)

    return ac

def Ols_regularizado(Xtreino,Ytreino,Xteste,Yteste):
    coef = np.linalg.pinv(Xtreino.T @ Xtreino + 0.1*np.identity(Xtreino.shape[1])) @ Xtreino.T @ Ytreino
 
    ac = acuracia(coef,Xteste,Yteste)

    return ac

Data = np.loadtxt('EMG.csv', delimiter=',')
Rotulos = np.loadtxt('Rotulos.csv', delimiter=',')
Rodadas = 100
ac_ols = []
ac_ols_regularizado = []

for i in range(Rodadas):
    seed = np.random.permutation(Data.shape[0])
    X = Data[seed,:]
    Y = Rotulos[seed,:]

    Xtreino = X[0:int(X.shape[0]*.8),:]
    Ytreino = Y[0:int(X.shape[0]*.8),:]

    Xteste = X[int(X.shape[0]*.8):,:]
    Yteste = Y[int(Y.shape[0]*.8):,:]

    Xtreino = np.concatenate((np.ones((Xtreino.shape[0],1)), Xtreino),axis=1)
    Xteste = np.concatenate((np.ones((Xteste.shape[0],1)), Xteste),axis=1)

    ac_ols.append(Ols(Xtreino,Ytreino,Xteste,Yteste))
    ac_ols_regularizado.append(Ols_regularizado(Xtreino,Ytreino,Xteste,Yteste))


print('Media: ', np.mean(ac_ols))
print('Desvio padrao: ', np.std(ac_ols))
print('Menor valor: ', np.min(ac_ols))
print('Maior valor: ', np.max(ac_ols))

print('----------------------------')

print('Media: ', np.mean(ac_ols_regularizado))
print('Desvio padrao: ', np.std(ac_ols_regularizado))
print('Menor valor: ', np.min(ac_ols_regularizado))
print('Maior valor: ', np.max(ac_ols_regularizado))

