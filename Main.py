# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:13:22 2017

@author: Administrator
"""
import pandas as pd
import TestAlls as TAS
import BuildingModel as BM
import Evaluate as E
import numpy as np
from sklearn.model_selection import StratifiedKFold 
import Data
from copy import deepcopy
datasetsPaths={#Windows 下的路径
               "ant":["bugs\\ant-1.3.csv",
                       "bugs\\ant-1.4.csv",
                       "bugs\\ant-1.5.csv",
                       "bugs\\ant-1.6.csv",
                       "bugs\\ant-1.7.csv"],
               "camel":["bugs\\camel-1.0.csv",
                       "bugs\\camel-1.2.csv",
                       "bugs\\camel-1.4.csv",
                       "bugs\\camel-1.6.csv"],
               "jedit":["bugs\\jedit-3.2.csv",
                       "bugs\\jedit-4.0.csv",
                       "bugs\\jedit-4.1.csv",
                       "bugs\\jedit-4.2.csv"],
               "synapse":["bugs\\synapse-1.0.csv",
                       "bugs\\synapse-1.1.csv",
                       "bugs\\synapse-1.2.csv"],
               "ivy":["bugs\\ivy-1.1.csv",
                      "bugs\\ivy-1.4.csv",
                      "bugs\\ivy-2.0.csv"],
               "xalan":["bugs\\xalan-2.4.csv",
                        "bugs\\xalan-2.5.csv",
                        "bugs\\xalan-2.6.csv"],
                "xerces":["bugs\\xerces-1.2.csv",
                          "bugs\\xerces-1.3.csv"]
               }
models=["EALR","BRR","DTR","NNR","GBR","RF","AdaBoostR2_LR","AdaBoostR2_DTR","AdaBoostR2_BRR"]
metrics=["FPA","Kendall"]
testSets=["ant-1.7","ant-1.6","ant-1.5","camel-1.6","camel-1.4","camel-1.2","ivy-2.0",
          "jedit-4.2","jedit-4.1","jedit-4.0","synapse-1.1","synapse-1.2",
          "xalan-2.5","xalan-2.6","xerces-1.3"]
trainingProjs=["ant","camel","jedit","synapse","ivy","xalan","xerces"]
unsupervisedModels=["U_wmc+","U_dit+","U_noc+","U_cbo+","U_rfc+","U_lcom+","U_ca+","U_ce+","U_npm+", "U_lcom3+","U_dam+",
                    "U_moa+","U_mfa+","U_cam+","U_ic+","U_cbm+","U_amc+","U_max_cc+","U_avg_cc+","U_loc+",
                    "U_wmc-","U_dit-","U_noc-","U_cbo-","U_rfc-","U_lcom-","U_ca-","U_ce-","U_npm-", "U_lcom3-","U_dam-",
                    "U_moa-","U_mfa-","U_cam-","U_ic-","U_cbm-","U_amc-","U_max_cc-","U_avg_cc-","U_loc-"]


#if you just run the program, you can only use the function f1 and f2 in file TestAlls


#Here is some examples:
#you can delete them and write some simple esamples
 
#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA            
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Within-Version",testSetName=i,trainingProjName=None)
    data=datasets.getTrainingAndTestSet()[0]
    y=pd.Series([None]*len(data))
    for i in range(len(data)):
        if(data.iloc[i,-1] == 0):
            y[i]="N"
        else:
            y[i]="Y"
    skf=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(X=data,y=y))
    t=None
    for i in range(5):#对于每一折
        trainingSet=deepcopy(data.iloc[skf[i][0],:])
        testSet=deepcopy(data.iloc[skf[i][1],:])    
        tf=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],rus=False,smoter=False,burak=False)
        if(t is None):
            t=tf
        else:
            t=t+tf
    t=t/5.0
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
        
        
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
df=pd.DataFrame(result,index=rows,columns=models)
df.to_csv("A-WithinVersion-Simple.csv")


#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB          
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Within-Version",testSetName=i,trainingProjName=None)
    data=datasets.getTrainingAndTestSet()[0]
    y=pd.Series([None]*len(data))
    for i in range(len(data)):
        if(data.iloc[i,-1] == 0):
            y[i]="N"
        else:
            y[i]="Y"
    skf=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(X=data,y=y))
    t=None
    for i in range(5):#对于每一折
        trainingSet=deepcopy(data.iloc[skf[i][0],:])
        testSet=deepcopy(data.iloc[skf[i][1],:])    
        tf=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=True,rus=False,burak=False)
        if(t is None):
            t=tf
        else:
            t=t+tf
    t=t/5.0
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
        
        
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
columns=["EALR+SMOTER","BRR+SMOTER","DTR+SMOTER","NNR+SMOTER","GBR+SMOTER","RF+SMOTER",
         "AdaBoostR2_LR+SMOTER","AdaBoostR2_DTR+SMOTER","AdaBoostR2_BRR+SMOTER"]
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("B-WithinVersion-SMOTER.csv")

#CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC         
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Within-Version",testSetName=i,trainingProjName=None)
    data=datasets.getTrainingAndTestSet()[0]
    y=pd.Series([None]*len(data))
    for i in range(len(data)):
        if(data.iloc[i,-1] == 0):
            y[i]="N"
        else:
            y[i]="Y"
    skf=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(X=data,y=y))
    t=None
    for i in range(5):#对于每一折
        trainingSet=deepcopy(data.iloc[skf[i][0],:])
        testSet=deepcopy(data.iloc[skf[i][1],:])    
        tf=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=True,burak=False)
        if(t is None):
            t=tf
        else:
            t=t+tf
    t=t/5.0
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
        
        
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
columns=["EALR+RUS","BRR+RUS","DTR+RUS","NNR+RUS","GBR+RUS","RF+RUS",
         "AdaBoostR2_LR+RUS","AdaBoostR2_DTR+RUS","AdaBoostR2_BRR+RUS"]
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("C-WithinVersion-RUS.csv")


        
        
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEWithinVersin 无监督
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Within-Version",testSetName=i,trainingProjName=None)
    data=datasets.getTrainingAndTestSet()[0]
    y=pd.Series([None]*len(data))
    for i in range(len(data)):
        if(data.iloc[i,-1] == 0):
            y[i]="N"
        else:
            y[i]="Y"
    skf=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(X=data,y=y))
    t=None
    for i in range(5):#对于每一折
        trainingSet=deepcopy(data.iloc[skf[i][0],:])
        testSet=deepcopy(data.iloc[skf[i][1],:]) 
        tmetrics=[]
        for j in ["FPA","Kendall"]: 
            tmetric=[]#存放每一行的度量值
            for k in unsupervisedModels:
                umd=BM.buildModel(trainingSet=testSet,modelType=k)
                tmetric.append(E.Eva(model=umd,testSet=testSet,metric=j,predValue="bug"))     
            tmetrics.append(tmetric)
        tmetrics=np.array(tmetrics)
        if(t is None):
            t=tmetrics
        else:
            t=t+tmetrics
    t=t/5.0
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
        
        
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
columns=["EALR+SMOTERDE","BRR+SMOTERDE","DTR+SMOTERDE","NNR+SMOTERDE","GBR+SMOTERDE","RF+SMOTERDE"]
df=pd.DataFrame(result,index=rows,columns=unsupervisedModels)
df.to_csv("E-WithinVersion-Unsupervised.csv")

#FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Version",testSetName=i,trainingProjName=None)
    trainingSet,testSet=datasets.getTrainingAndTestSet()
    t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=False,burak=False)
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
df=pd.DataFrame(result,index=rows,columns=models)
df.to_csv("F-CrossVersion-Simple.csv")

#GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Version",testSetName=i,trainingProjName=None)
    trainingSet,testSet=datasets.getTrainingAndTestSet()
    t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=True,rus=False,burak=False)
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
columns=["EALR+SMOTER","BRR+SMOTER","DTR+SMOTER","NNR+SMOTER","GBR+SMOTER","RF+SMOTER",
         "AdaBoostR2_LR+SMOTER","AdaBoostR2_DTR+SMOTER","AdaBoostR2_BRR+SMOTER"]
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("G-CrossVersion-SMOTER.csv")
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
result=None
for i in testSets:  
    datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Version",testSetName=i,trainingProjName=None)
    trainingSet,testSet=datasets.getTrainingAndTestSet()
    t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=True,burak=False)
    if(result is None):
        result=t
    else:
        result=np.vstack([result,t])
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
columns=["EALR+RUS","BRR+RUS","DTR+RUS","NNR+RUS","GBR+RUS","RF+RUS",
         "AdaBoostR2_LR+RUS","AdaBoostR2_DTR+RUS","AdaBoostR2_BRR+RUS"]
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("H-CrossVersion-Rus.csv")

#JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ无监督模型
result=None
for i in testSets:
    datasets=Data.Data(paths=datasetsPaths,scenario="Within-Version",testSetName=i,trainingProjName=None)
    testSet=datasets.getTrainingAndTestSet()[0]
    for j in ["FPA","Kendall"]:
        t=[]
        for k in unsupervisedModels:
            umd=BM.buildModel(trainingSet=testSet,modelType=k)
            t.append(E.Eva(model=umd,testSet=testSet,metric=j,predValue="bug"))
        t=np.array(t)
        if(result is None):
            result=t
        else:
            result=np.vstack([result,t])        
rows=['ant-1.7<=>FPA', 'ant-1.7<=>Kendall', 'ant-1.6<=>FPA', 'ant-1.6<=>Kendall', 'ant-1.5<=>FPA', 'ant-1.5<=>Kendall', 
      'camel-1.6<=>FPA', 'camel-1.6<=>Kendall', 'camel-1.4<=>FPA', 'camel-1.4<=>Kendall', 'camel-1.2<=>FPA', 'camel-1.2<=>Kendall', 
      'ivy-2.0<=>FPA', 'ivy-2.0<=>Kendall', 'jedit-4.2<=>FPA', 'jedit-4.2<=>Kendall', 'jedit-4.1<=>FPA', 'jedit-4.1<=>Kendall', 
      'jedit-4.0<=>FPA', 'jedit-4.0<=>Kendall', 'synapse-1.1<=>FPA', 'synapse-1.1<=>Kendall', 'synapse-1.2<=>FPA', 
      'synapse-1.2<=>Kendall', 'xalan-2.5<=>FPA', 'xalan-2.5<=>Kendall', 'xalan-2.6<=>FPA', 'xalan-2.6<=>Kendall', 
      'xerces-1.3<=>FPA', 'xerces-1.3<=>Kendall']
df=pd.DataFrame(result,index=rows,columns=unsupervisedModels)
df.to_csv("J-Unsupervised.csv")

#KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
result=None
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Project",testSetName=i,trainingProjName=j)
        trainingSet,testSet=datasets.getTrainingAndTestSet()
        t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=False,burak=False)
        if(result is None):
            result=t
        else:
            result=np.vstack([result,t])
rows=[]
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        rows.append(i+"<=>"+j+"FPA")
        rows.append(i+"<=>"+j+"Kendall")
df=pd.DataFrame(result,index=rows,columns=models)
df.to_csv("K-CrossProject-Simple.csv")


#LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
result=None
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Project",testSetName=i,trainingProjName=j)
        trainingSet=pd.read_csv("bugs\\burak\\"+i+"--"+j+".csv")
        testSet=datasets.getTrainingAndTestSet()[1]
        t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=False,burak=False)
        if(result is None):
            result=t
        else:
            result=np.vstack([result,t])
rows=[]
columns=["EALR+Burak","BRR+Burak","DTR+Burak","NNR+Burak","GBR+Burak","RF+Burak",
         "AdaBoostR2_LR+Burak","AdaBoostR2_DTR+Burak","AdaBoostR2_BRR+Burak"]
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        rows.append(i+"<=>"+j+"FPA")
        rows.append(i+"<=>"+j+"Kendall")
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("L-CrossProject-Burak.csv")

#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
result=None
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Project",testSetName=i,trainingProjName=j)
        trainingSet=pd.read_csv("bugs\\burak\\"+i+"--"+j+".csv")
        testSet=datasets.getTrainingAndTestSet()[1]
        t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=True,rus=False,burak=False)
        if(result is None):
            result=t
        else:
            result=np.vstack([result,t])
rows=[]
columns=["EALR+Burak+SMOTER","BRR+Burak+SMOTER","DTR+Burak+SMOTER","NNR+Burak+SMOTER","GBR+Burak+SMOTER","RF+Burak+SMOTER",
         "AdaBoostR2_LR+Burak+SMOTER","AdaBoostR2_DTR+Burak+SMOTER","AdaBoostR2_BRR+Burak+SMOTER"]
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        rows.append(i+"<=>"+j+"FPA")
        rows.append(i+"<=>"+j+"Kendall")
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("M-CrossProject-Burak-SMOTER.csv")

#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
result=None
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Project",testSetName=i,trainingProjName=j)
        trainingSet=pd.read_csv("bugs\\burak\\"+i+"--"+j+".csv")
        testSet=datasets.getTrainingAndTestSet()[1]
        t=TAS.f1(trainingSet=trainingSet,testSet=testSet,modelNames=models,metrics=["FPA","Kendall"],smoter=False,rus=True,burak=False)
        if(result is None):
            result=t
        else:
            result=np.vstack([result,t])
rows=[]
columns=["EALR+Burak+Rus","BRR+Burak+Rus","DTR+Burak+Rus","NNR+Burak+Rus","GBR+Burak+Rus","RF+Burak+Rus",
         "AdaBoostR2_LR+Burak+Rus","AdaBoostR2_DTR+Burak+Rus","AdaBoostR2_BRR+Burak+Rus"]
for i in testSets:
    for j in trainingProjs:
        if(j in i):
            continue
        rows.append(i+"<=>"+j+"FPA")
        rows.append(i+"<=>"+j+"Kendall")
df=pd.DataFrame(result,index=rows,columns=columns)
df.to_csv("N-CrossProject-Burak-Rus.csv")





#生成burak的原始数据集
#for i in testSets:
#    for j in trainingProjs:
#        if(j in i):
#            continue
#        print(i+"<=>"+j+"......")
#        datasets=Data.Data(paths=datasetsPaths,scenario="Cross-Project",testSetName=i,trainingProjName=j)
#        trainingSet,testSet=datasets.getTrainingAndTestSet()
#        dpp=DataPreProcessing(data=trainingSet,targetData=testSet,burakKNN=10,burak=True,smoter=False,rusRatio=75,
#              rus=False,Drange=[list(range(5,21)),[0,1,2,3,4,5,6],(0.1,5)],metric=None,modelName=None,
#              F=0.7,CR=0.3,PopulationSize=10,Lives=8,smoterde=False)
#        newTrainingSet=dpp.preProcess()
#        newTrainingSet.to_csv("D:\\Burak\\"+i+"--"+j+".csv",index=False)




















