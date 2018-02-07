# -*- coding: utf-8 -*-

"""
@author: Zhdun
"""
import Evaluate as E
from SMOTER import SMOTER
from DE import DE
from sklearn.model_selection import StratifiedKFold   
from copy import deepcopy
import pandas as pd
import BuildingModel as BM
class SMOTERDE:
    """
    基于差分进化的SMOTER
    直观上讲存在一定的针对性，即针对特定数据集、特定指标和特定模型的优化
    """
    def __init__(self,data,Drange,metric,modelName,F=0.7,CR=0.3,PopulationSize=10,Lives=8):
        self.data=data
        self.Drange=Drange
        self.metric=metric
        self.modelName=modelName
        self.F=F
        self.CR=CR
        self.PopulationSize=PopulationSize
        self.Lives=Lives
    def stratify(self,K,dataSet,randomSeed):
        """
        分层，返回训练集和测试集的元祖
        """
        # In[]为了分层，先根据分类属性生成一个二值序列y
        y=pd.Series([None]*dataSet.shape[0])
        for i in range(dataSet.shape[0]):
            if(dataSet.iloc[i,-1] == 0):
                y[i]="N"
            else:
                y[i]="Y"
        # In[] 分层分组
        skf=StratifiedKFold(n_splits=K,shuffle=True,random_state=randomSeed)
        return list(skf.split(X=dataSet,y=y))    
    def fitness(self,targetPars,otherPars):
        """
        对于SMOTUNED，otherPars有一个值，数据集
        """
        res=[]
        data=deepcopy( otherPars[0])
        stf=self.stratify(K=3,dataSet=data,randomSeed=1)
        for i in range(3):
            trainingSet=deepcopy(data.iloc[stf[i][0],:])
            testSet=deepcopy(data.iloc[stf[i][1],:])
            S=SMOTER(targetPars[0,0],targetPars[0,1],targetPars[0,2],trainingSet)
            trainingSet_SMOTE=S.smoteR()
            model=BM.buildModel(trainingSet=trainingSet_SMOTE,modelType=self.modelName)
            res.append( E.Eva(model,testSet=testSet,metric=self.metric,predValue= "bug"))
        return sum(res)/len(res)
        
    def getData(self):
        """
        获取进化后的数据集
        """
        #Setep1 获取差分进化后SMOTER参数
        de=DE(fitness=self.fitness,D=3,DRange=self.Drange,F=self.F,CR=self.CR,PopulationSize=self.PopulationSize,Lives=self.Lives)
        bestPars= de.evolution(otherPars=[self.data])
        #Step2 使用最佳参数获得数据集
        S=SMOTER(k=bestPars[0,0],m=bestPars[0,1],r=bestPars[0,2],data=self.data)
        return (S.smoteR(),bestPars)
               
        
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        