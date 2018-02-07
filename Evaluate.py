# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:10:59 2017

@author: Administrator
"""
from copy import deepcopy
import numpy as np
#import DataPreProcessing as DPP
def AUL(x,y):
    """
    计算折线下的面积,直接连线的方式
    
    Parameters
    ----------
    x:横坐标集合, array-like
    
    y:纵坐标集合, array-like
    
    Returns
    ------
    折线下的面积
    """
    area=0
    for i in range(1,len(x)):
        area = area + (y[i] + y[i-1])*(x[i]-x[i-1])
    return area/2        

def popt(model,testSet,predValue="density"):
    """
    计算popt指标,Model 直接预测缺陷密度
    Parameter:
    ----------
    
    model:训练好的模型
    
    testSet：测试集，要有loc和bug
    
    predValue:模型预测的是bug还是密度,bug,density
    
    """ 
    
    loc=deepcopy(list(testSet.loc[:,"loc"]))
    bug=deepcopy(list(testSet.loc[:,"bug"]))
    if(predValue=="density"):
        cols=list(testSet.columns)
        cols.remove("bug")
        cols.remove("loc")
        ts=testSet.loc[:,cols]
    elif(predValue=="bug"):
        cols=list(testSet.columns) 
        cols.remove("bug")
        ts=testSet.loc[:,cols]        
    predictResult = list(model.predict(ts))
    t=list(zip(loc,bug,predictResult))
        # In[]计算Optimal Area  
    t.sort(key = lambda arg : arg[1] / arg[0] if arg[0] != 0 else 0, reverse = True)#按缺陷密度降序排序
    x , y = [0] , [0]
    for i in range(0,len(t)):
        x.append(x[i]+t[i][0])
        y.append(y[i]+t[i][1])        
    areaOpt = AUL(x,y) 
    # In[]计算Worst Area  
    t.reverse()
    x , y = [0] , [0]
    for i in range(0,len(t)):
        x.append(x[i]+t[i][0])
        y.append(y[i]+t[i][1])        
    areaWorst = AUL(x,y)   
    # In[]计算our model Area  
    if(predValue=="density"):
        t.sort(key = lambda arg : arg[2], reverse = True)
    elif(predValue=="bug"):
        t.sort(key = lambda arg : arg[2] / arg[0] if arg[0] != 0 else 0, reverse = True)
    x , y = [0] , [0]
    for i in range(0,len(t)):
        x.append(x[i]+t[i][0])
        y.append(y[i]+t[i][1])        
    areaM = AUL(x,y)  
    # In[] 返回Popt
    return (areaM - areaWorst) / (areaOpt - areaWorst ) 
def Kendall(model,testSet,predValue="density"):
    """
    计算Kendall,越接近0代表两组变量相关性越小，模型越接近于垃圾随机模型
    Parameter:
    ----------
    
    model:训练好的模型
    
    testSet：测试集，要有loc和bug
    
    predValue:模型预测的是bug还是密度,bug,density
    
    """ 
    #获取预测结果
    loc=deepcopy(list(testSet.loc[:,"loc"]))
    bug=deepcopy(list(testSet.loc[:,"bug"]))
    if(predValue=="density"):
        cols=list(testSet.columns)
        cols.remove("bug")
        cols.remove("loc")
        ts=testSet.loc[:,cols]
    elif(predValue=="bug"):
        cols=list(testSet.columns) 
        cols.remove("bug")
        ts=testSet.loc[:,cols]        
    X = list(model.predict(ts))
    #组队
    if(predValue=="bug"):
        Y=bug
    elif(predValue=="density"):
        Y=[]
        for i,j in zip(bug,loc):
            if(j==0):
                Y.append(0)
            else:
                Y.append(i/j)
       
    # 开始计算
    n,n1,n2=len(X),0,0
    
    for i in range(n):
        for j in range(i+1,n):
            if(X[i]>X[j] and Y[i]>Y[j]):
                n1=n1+1
            elif(X[i]<X[j] and Y[i]<Y[j]):
                n1=n1+1
            elif(X[i]>X[j] and Y[i]<Y[j]):
                n2=n2+1
            elif(X[i]<X[j] and Y[i]>Y[j]):
                n2=n2+1

    return (2*(n1-n2))/(n*(n-1))

def FPA(model,testSet,predValue="bug"):
    """
    计算FPA，必须是预测bug数目的模型
    Parameter:
    ----------
    
    model:训练好的模型
    
    testSet：测试集，要有loc和bug
    
    predValue:模型预测的是bug还是密度,bug,density
    
    """ 
    #获取预测结果
    if(predValue != "bug"):
        return 0;
    bug=deepcopy(list(testSet.loc[:,"bug"]))
    k=len(bug);
    n=sum(bug);
    cols=list(testSet.columns) 
    cols.remove("bug")
    ts=testSet.loc[:,cols]        
    predResults = list(model.predict(ts))
    t=list(zip(bug,predResults))
    t.sort(key = lambda arg : arg[1])#按预测结果升序排序
    fpa=0
    for m in range(1,k+1):
        tvalue=0
        for i in range(k-m,k):
            tvalue=tvalue+t[int(i)][0]
        fpa=fpa+(tvalue/n)
    return fpa/k;
            
    
    
# In[] 计算Acc_20
def Acc_20(model,testSet,predValue="bugs"):
    loc=deepcopy(list(testSet.loc[:,"loc"]))
    bug=deepcopy(list(testSet.loc[:,"bug"]))
    if(predValue=="density"):
        cols=list(testSet.columns)
        cols.remove("loc")
        cols.remove("bug")
        ts=testSet.loc[:,cols]
    elif(predValue=="bug"):
        cols=list(testSet.columns) 
        cols.remove("bug")
        ts=testSet.loc[:,cols]        
    predictResult = list(model.predict(ts))
    t=list(zip(loc,bug,predictResult))
    if(predValue=="density"):
        t.sort(key = lambda arg : arg[2], reverse = True)
    elif(predValue=="bug"):
        t.sort(key = lambda arg : arg[2] / arg[0] if arg[0] != 0 else 0, reverse = True)
        
    totalLocs,totalBugs,bugs20,locs20=sum(loc),sum(bug),0.0,0.0
    threshold = totalLocs*0.2;
    for i in range(0,len(t)):
        if(locs20>=threshold):
            break
        bugs20=bugs20+t[i][1]
        locs20=locs20+t[i][0]
    return bugs20/totalBugs

def F1(model,testSet):
    predRes=list(model.predict(testSet.iloc[:,range(0,testSet.shape[1]-1)]))
    trueRes=list(testSet.iloc[:,-1])
    # TP  真实有缺陷，预测有缺陷
    # FN 真实有缺陷，预测无缺陷
    # FP 真实无缺陷，预测为有缺陷
    # TN 真实无缺陷，预测为无缺陷
    TP,FP,TN,FN=0,0,0,0
    for i in range(0,len(predRes)):
        if(trueRes[i]=='Y' and predRes[i]=='Y'):
            TP=TP+1;
        elif(trueRes[i]=='Y' and predRes[i]=='N'):
            FN=FN+1
        elif(trueRes[i]=='N' and predRes[i]=='Y'):
            FP=FP+1
        elif(trueRes[i]=='N' and predRes[i]=='N'):
            TN=TN+1
    tDict={"真实有缺陷，预测有缺陷":TP,"真实有缺陷，预测无缺陷":FN,
           "真实无缺陷，预测有缺陷":FP,"真实无缺陷，预测无缺陷":TN}
    return tDict
    
    
def ACC(model,testSet,predValue="bug"):
    """
    计算准确率，原则上只计算预测的bug数目
    """
    if(predValue!="bug"):
        return 0
    cols=list(testSet.columns) 
    cols.remove("bug")
    predRes=list(model.predict(testSet.loc[:,cols]))
    trueRes=list(testSet.iloc[:,-1])
    t=0
    for i in range(len(predRes)):
        if(int(predRes[i])==int(trueRes[i])):
            t=t+1
    return t/len(trueRes)
   
def Eva(model,testSet,metric,predValue):
    if(metric=="popt"):
        return popt(model,testSet,predValue)
    elif(metric=="Acc20"):
        return Acc_20(model,testSet,predValue)
    elif(metric=="Kendall"):
        return Kendall(model,testSet,predValue)
    elif(metric=="FPA"):
        return FPA(model,testSet,predValue)

#def Validation(models,testSet,metrics):
#    res=[]
#    for metric in metrics:
#        t=[]
#        for model in models:
#            t.append(Eva(model,testSet,metric,"bug"))
#        res.append(t)
#    return np.array(res)




        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


