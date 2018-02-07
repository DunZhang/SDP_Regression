# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:34:14 2017

@author: Zhdun
"""
from copy import deepcopy
import numpy as np
from SMOTERDE import SMOTERDE
import SMOTER
import random
class DataPreProcessing:
    """
    用来对数据集做各种处理，比如SMOTE、布鲁克过滤法等。
    """
    def __init__(self,data,targetData,burakKNN,burak,smoter,rusRatio,rus,Drange,metric,modelName,F,CR,PopulationSize,Lives,smoterde):
        
        self.data=deepcopy(data)#原始数据
        
        
        self.targetData=deepcopy(targetData)#目标数据集，使用布鲁克过滤法会用到
        self.burakKNN=burakKNN#布鲁克过滤法中，通过一个目标实例选择burakKNN个实例
        self.burak=burak#是否使用布鲁克过滤法
        
        
        
        self.smoter=smoter#是否使用smote过采样
        
        self.rusRatio=rusRatio#使用欠随机采样时，多数类留下来的比例，[0,100]
        self.rus=rus#是否使用随机欠采样
        
        self.Drange=Drange
        self.metric=metric
        self.modelName=modelName
        self.F=F
        self.CR=CR
        self.PopulationSize=PopulationSize
        self.Lives=Lives
        self.bestParas=None
        self.smoterde=smoterde#是否使用差分进化smote过采样方法
        
        
    def __burak(self):
        trainingSet=self.data.iloc[:,0:self.data.shape[1]-1]
        testSet=self.targetData.iloc[:,0:self.targetData.shape[1]-1]
        trainingSetIndexs=set()
        eucs = np.zeros((trainingSet.shape[0],testSet.shape[0]))
        trSet=np.array(trainingSet)
        teSet=np.array(testSet)
        for col in range(0,trainingSet.shape[0]):
            for row in range(0,testSet.shape[0]):
                eucs[col,row]=np.linalg.norm(trSet[col,:]-teSet[row,:])
        for col in range(0,eucs.shape[1]):#对于每个测试集实例
            for i in range(0,self.burakKNN):
                t=eucs[:,col].argmin()
                trainingSetIndexs.add(t)
                eucs[t,col]=float("inf")
        self.data=(deepcopy(self.data).iloc[list(trainingSetIndexs),:]).reset_index(drop=True)
        
    def __rus(self):
        d=deepcopy(self.data)
        PIndexs,NIndexs=[],[]
        for i in range(len(d)):
            if(d.iloc[i,-1]==0):
                NIndexs.append(i)
            else:
                PIndexs.append(i)
        if(self.rusRatio is None):
            self.rusRatio=100.0*len(PIndexs)/len(NIndexs)
        NIndexs=random.sample(NIndexs,int(self.rusRatio/100.0*len(NIndexs)))
        NIndexs=PIndexs+NIndexs
        self.data= d.iloc[NIndexs,:].reset_index(drop=True)
        #######################################################
#        td=deepcopy(self.data)
#        t1=len(td[td['bug']==0])
#        print(t1,len(td))
        #########################################################
    
    def __SMOTER(self):
        s=SMOTER.SMOTER(k=5,m=6,r=2,data=self.data)
        self.data=s.smoteR()
    
    def __SMOTERDE(self):
        su=SMOTERDE(data=self.data,Drange=self.Drange,metric=self.metric,modelName=self.modelName,F=self.F,
                    CR=self.CR,PopulationSize=self.PopulationSize,Lives=self.Lives)
        self.data,self.bestParas= su.getData()
                
    def preProcess(self):
        if(self.burak):
            self.__burak()
        if(self.rus):
            self.__rus()
        elif(self.smoter):
            self.__SMOTER()
        elif(self.smoterde):
            self.__SMOTERDE()
        return self.data
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        