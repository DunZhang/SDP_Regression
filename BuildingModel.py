# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:47:39 2017

@author: Administrator
"""
from sklearn import linear_model
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import UnsupervisedModel
def buildModel(trainingSet,modelType):
    """
    构建模型
    
    Parameters
    ----------
    trainingSet : 训练集, array-like
    
    modelType :模型类型 LR(Linear Regression),BRR(Bayesian Ridge Regression),SVR(Support Vector Regression),
    NNR(Nearest Neighbours Regression),DTR(Decision Tree Regression),GBR(GGradient Boosting Regression)
    
    Returns
    -------
    model
    """
    # In[] 建立模型
    model=None
    if(modelType == "EALR"):
        model = linear_model.LinearRegression()
    elif(modelType == "BRR"):
        model = linear_model.BayesianRidge(compute_score=True)
    elif(modelType == "SVR"):
        model = SVR()
    elif(modelType == "NNR"):
        model = KNeighborsRegressor()
    elif(modelType == "DTR"):
        model = DecisionTreeRegressor()
    elif(modelType == "GBR"):
        model = GradientBoostingRegressor() 
    elif(modelType=="RF"):
        model=RandomForestRegressor()
    elif("U_" in modelType):#无监督算法
        model=UnsupervisedModel.UnsupervisedModel(label=modelType[2:])
    elif("AdaBoostR2" in modelType):
        base_m=modelType.split("_")[1]
        if(base_m=="LR"):
            model=AdaBoostRegressor(base_estimator= linear_model.LinearRegression())
        elif(base_m=="DTR"):
            model=AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
        elif(base_m=="BRR"):
            model=AdaBoostRegressor(base_estimator=linear_model.BayesianRidge(compute_score=True))
        
    # In[] 训练模型    
    model.fit(X = trainingSet.iloc[:,range(0,trainingSet.shape[1]-1)], y = trainingSet.iloc[:,-1])
    return model
        
        
        
