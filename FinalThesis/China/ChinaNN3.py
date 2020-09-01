
import pandas as pd
import numpy as np
import math as ma
import matplotlib.pyplot as plt

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from sklearn import preprocessing

from sklearn.preprocessing import normalize

min_max_scaler = preprocessing.MinMaxScaler()

Cpro=pd.read_csv("Cpro.csv")
Cval=pd.read_csv("Cval.csv")
Ctra=pd.read_csv("Ctra.csv")
Cmom=pd.read_csv("Cmom.csv")
Cfd=pd.read_csv("Cfd.csv")
Ctec=pd.read_csv("Ctec.csv")

def fnn3(frame,string):
    df=frame.dropna(axis=0,how='any')
    name=string
    nm1=name+"tst2.png"
    nm2=name+"cov2.png"
    x=df.drop(columns=['code','date','price'])
    x=np.array(x)
    x=normalize(x, axis=0, norm='max')
    y=np.array(df['ret'])
    x=np.delete(x,1,axis=1)
    xdim=x.shape[1]
    ydim=1
    DS=SupervisedDataSet(xdim,ydim)
    
    for i in range(len(x)):
        DS.addSample(x[i],y[i])
    
    dataTrain, dataTest = DS.splitWithProportion(0.8)
    dataPlot, datadrop =DS.splitWithProportion(0.08)
    xTrain, yTrain = dataTrain['input'],dataTrain['target']
    xTest, yTest = dataTest['input'], dataTest['target']
    xPlot, yPlot= dataPlot['input'], dataPlot['target']
    fnn=buildNetwork(xdim,xdim+1,xdim+2,int(0.5*(xdim+1)),ydim,hiddenclass=TanhLayer,outclass=LinearLayer)
    trainer=BackpropTrainer(fnn,dataTrain,learningrate=0.0001,verbose=True)
    err_train, err_valid =trainer.trainUntilConvergence(maxEpochs=100)
    
    predict_resutl=[]
    for i in np.arange(len(xPlot)):
        predict_resutl.append(fnn.activate(xPlot[i])[0])
    print(predict_resutl)
    
    tstresult = percentError( trainer.testOnClassData(), dataTest['target'] )
    print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tstresult)
    
   #yTest2=yTest([0:len(yTest):12])
   #pred2=predict_resutl([0:len(predict_resutl):12])
    
    plt.figure(figsize=(30,6), dpi=600)
    plt.xlabel("Test Timeline")
    plt.ylabel("Result")
    plt.plot(np.arange(0,len(xPlot)), yPlot,'ko-', label='true number')
    plt.plot(np.arange(0,len(xPlot)), predict_resutl,'ro--', label='predict number')
    lgnd1=plt.legend()
    plt.savefig(nm1,dpi=600, bbox_extra_artists=(lgnd1))
    
    plt.figure(figsize=(9,9), dpi=600)
    plt.plot(err_train,'b',label='train_err')
    plt.plot(err_valid,'r',label='valid_err')
    lgnd2=plt.legend()
    plt.xlabel("Training Times")
    plt.ylabel("Total Error")
    plt.savefig(nm2,dpi=600, bbox_extra_artists=(lgnd2))
    plt.show()
    
    return
  
fnn3(Cpro,"Cpro")
fnn3(Cval,"Cval")
fnn3(Ctra,"Ctra")
fnn3(Cmom,"Cmom")
fnn3(Cfd,"Cfd")
fnn3(Ctec,"Ctec")

