# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:48:13 2017

@author: Zhdun
"""
import pandas as pd
class Data:
    """
    用来获取训练集和测试集
    """
    def __init__(self,paths,scenario,testSetName,trainingProjName):
        """
        Parameters:
        ----------
        paths:
            文件路径集合
            
        scenario:
            应用场景，Within-Version，Cross-Version，Cross-Project
            
        testSetName:
            测试集名称
            
        trainingProjName:
            训练集名称
        """
        self.paths=paths
        self.scenario=scenario
        self.testSetName=testSetName
        self.trainingProjName=trainingProjName
    def __getPaths(self):  
        """
        获取训练集路径和测试集路径集合
        Return:
        -------
        (训练集路径集合，测试集路径集合)
        """
        trainingSetsPaths,testSetsPaths=[],[]
        # In[]获取测试集路径
        for i in self.paths[self.testSetName.split("-")[0]]:
            if(self.testSetName in i):
                testSetsPaths.append(i)
                break
        # In[]获取训练集
        if(self.scenario=="Within-Version"):
            trainingSetsPaths=testSetsPaths
        elif(self.scenario=="Cross-Version"):
            for i in self.paths[self.testSetName.split("-")[0]]:
                if(self.testSetName in i):
                    break
                trainingSetsPaths.append(i)
        elif(self.scenario=="Cross-Project"):
            trainingSetsPaths=self.paths[self.trainingProjName]
        return (trainingSetsPaths,testSetsPaths)
    
    def __getDataSets(self,paths):
        data=None
        for i in paths:
            if(data is None):
                data=pd.read_csv(i)
            else:
                data=data.append(pd.read_csv(i))
        return data.reset_index(drop=True)
    
    def getTrainingAndTestSet(self):
        """
        返回未经处理直接读取的训练集和测试集
        """
        ps=self.__getPaths()
        return(self.__getDataSets(ps[0]),self.__getDataSets(ps[1]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    