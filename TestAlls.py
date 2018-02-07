# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:10:56 2017

@author: Administrator
"""
import BuildingModel as BM
import Evaluate as E
#from imblearn.over_sampling import SMOTE 

import numpy as np



from DataPreProcess import DataPreProcessing




def f1(trainingSet,testSet,modelNames,metrics,burak,smoter,rus):
    """
    不适用与SMOTERDE的测试
    Not suitable to SMOTEND+DE
    On given trainingSer and testSet, we return the metrcics' values for each models
    
    @Parameters:
        ---------------------------
    trainingSet:
        train model
        ----------------------------------
        
    testSet:
        test set
        ---------------------------------
        
    modelNames:
        list-like, store each models' names
        ----------------------------------
    
    metrics:
        measures, lisst-like
        -------------------------------------
        
    burak:
        whether use burak
        -------------------------------------
    
    smoter:
        whether use smoter
        --------------------------------------
        
    rus:
        whether use rus
        -------------------------------------------
    """
    res=[]
    ms=[]
    dpp=DataPreProcessing(data=trainingSet,targetData=testSet,burakKNN=10,burak=burak,smoter=smoter,rusRatio=None,
                  rus=rus,Drange=[list(range(5,21)),[0,1,2,3,4,5,6],(0.1,5)],metric=None,modelName=None,
                  F=0.7,CR=0.3,PopulationSize=10,Lives=8,smoterde=False)
    newTrainingSet=dpp.preProcess()
    for i in modelNames:#建立好所有模型
        ms.append(BM.buildModel(trainingSet=newTrainingSet,modelType=i))
    
    for metric in metrics:
        t=[]
        for model in ms:
            t.append(E.Eva(model=model,testSet=testSet,metric=metric,predValue="bug"))
        res.append(t)
    return np.array(res)

def f2(trainingSet,testSet,modelNames,metrics,burak):
    """
    不适用与SMOTERDE的测试
    Only suitable to SMOTEND+DE
    On given trainingSer and testSet, we return the metrcics' values for each models
    
    @Parameters:
        ---------------------------
    trainingSet:
        train model
        ----------------------------------
        
    testSet:
        test set
        ---------------------------------
        
    modelNames:
        list-like, store each models' names
        ----------------------------------
    
    metrics:
        measures, lisst-like
        -------------------------------------
        
    burak:
        whether use burak
        -------------------------------------
    
    smoter:
        whether use smoter
        --------------------------------------
        
    rus:
        whether use rus
        -------------------------------------------
    """
    metricValues,metricParas=[],[]
    for metric in metrics:
        modelValues,modelParas=[],[]#存放每一个模型的值,存放每个模型的最佳参数
        for model in modelNames:
            maxValue,bestParas=-10.0,None
            for ti in range(4):
                dpp=DataPreProcessing(data=trainingSet,targetData=testSet,burakKNN=10,burak=burak,smoter=False,rusRatio=75,
                              rus=False,Drange=[list(range(1,21)),[0,1,2,3,4,5,6],(0.1,5)],metric=metric,modelName=model,
                              F=0.7,CR=0.3,PopulationSize=10,Lives=8,smoterde=True)
                newTrainingSet=dpp.preProcess()
                mo=BM.buildModel(trainingSet=newTrainingSet,modelType=model)
                t=E.Eva(model=mo,testSet=testSet,metric=metric,predValue="bug")
                if(maxValue<t):
                    maxValue=t
                    bestParas=dpp.bestParas
                    bestParas=str(bestParas[0][0])+","+str(bestParas[0][1])+","+str(bestParas[0][2])
#                maxValue=2
#                bestParas="2,2,3"
            #这里的maxValue就是针对特定 训练集、测试集、指标和模型四个维度的一个特定值
            modelValues.append(maxValue)
            modelParas.append(bestParas)
        metricValues.append(modelValues)
        metricParas.append(modelParas)
    return (np.array(metricValues),np.array(metricParas))
               
            
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        