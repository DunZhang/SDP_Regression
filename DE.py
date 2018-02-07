# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:52:55 2017

@author: Zhdun
"""
import numpy as np
import random
from copy import deepcopy
from math import exp
class DE:
    """
    实现差分进化算法
    """
    def __init__(self,fitness,D,DRange,F=0.7,CR=0.3,PopulationSize=10,Lives=8):
        """
        parameters:
        ----------
        fiteness:
            适应度函数，计算函数的适应度值，形式 fitness(目标参数列表(1*n),其他参数列表)
            目标参数是指用来参与进化的参数，其个数就是染色体的基因数，1*n维向量
            其他参数是指适应度函数不参与进化的参数，python内置列表类型，比如训练集和测试集
        
        D:
            每个个体的维数，或者是每个染色体的基因
            
        DRange:
            list类型，每一维或基因的范围，范围可以是离散的或连续的
            
        F:
            缩放因子
            
        CR:
            交叉概率
            
        PopulationSize:
            种群大小
        
        Lives:
            进化次数
        """
        self.F,self.CR,self.PopulationSize,self.Lives,self.D,self.DRange=F,CR,PopulationSize,Lives,D,DRange
        self.fitnessValues=[]
        self.fitness=fitness
        self.bestValue=0#最佳适应度值
        self.bestIndex=0#有最佳适应度值的索引
        self.F0=0.5
    def __InitPop(self,otherPars):
        """
        初始化种群
        """
        pop=np.zeros((self.PopulationSize,self.D))#初始种群
        for i in range(pop.shape[0]):
            for j in range(self.D):
                if(type(self.DRange[j])==list):#离散
                    pop[i,j]=self.DRange[j][random.randint(0,len(self.DRange[j])-1)]
                else:#连续
                    pop[i,j]=random.uniform(self.DRange[j][0],self.DRange[j][1])
            self.fitnessValues.append(self.fitness(pop[i].reshape((1,self.D)),otherPars))
        self.pop=pop
        self.bestValue=max(self.fitnessValues)#当前最佳适应度值
        self.bestIndex=self.fitnessValues.index(self.bestValue)#当前最佳适应度值的染色体索引
    def __normalize(self,d):
        """将染色体中不符合要求的基因重新生成,注意!该函数不具有通用性"""
        v=deepcopy(d)
        v[0,0]=int(v[0,0]+0.5)
        v[0,1]=int(v[0,1]+0.5)
        for i in range(v.shape[1]):
            if(v[0,i]<self.DRange[i][0]):
                v[0,i]=self.DRange[i][0]
            elif(v[0,i]>self.DRange[i][-1]):
                v[0,i]=self.DRange[i][-1]
        return v
        
    def evolution(self,otherPars):
        """
        返回最佳个体基因
        """
        self.__InitPop(otherPars)
#        outT=np.array( deepcopy(self.fitnessValues))
#        outT=outT.reshape((self.PopulationSize,1))
#        print(np.hstack([self.pop,outT]))
#        print(self.bestValue)
#        print("=================================================================")
        for i in range(self.Lives):#进化lives代
#            print("第几代",i)
            tmp=np.zeros((self.PopulationSize,self.D))
#            self.F=self.F0* (  2**exp(1- self.Lives/(self.Lives-i)  ) )
            for j in range(self.pop.shape[0]):#对每一个目标向量
                old=self.pop[j].reshape((1,self.D))#目标向量
                new=np.zeros((1,self.D))#新生成的向量
                #变异操作
                indexs=list(range(self.pop.shape[0]))#索引集合
                indexs.remove(j)
                t=random.sample(indexs,3)
                x=self.pop[t[0]].reshape((1,self.D))
                y=self.pop[t[1]].reshape((1,self.D))
                z=self.pop[t[2]].reshape((1,self.D))
                v=self.__normalize( z+self.F*(x-y))#变异个体同时检查是否满足边界条件
                #交叉操作
                for k in range(self.D):
                    krand=random.randint(0,self.D-1)
                    if(random.random()<self.CR or k==krand ):
                        new[0,k]=v[0,k]
                    else:
                        new[0,k]=old[0,k]
                #选择操作，决定保留新向量还是旧向量,同时更新相关数组
                newValue=self.fitness(new,otherPars)
                if(newValue>self.fitnessValues[j]):#新向量好
                    self.fitnessValues[j]=newValue
                    if(self.bestValue<newValue):
                        self.bestValue=newValue
                        self.bestIndex=j
                        
                    tmp[j]=new
                else:
                    tmp[j]=old
            self.pop=tmp
#            outT=np.array( deepcopy(self.fitnessValues))
#            outT=outT.reshape((self.PopulationSize,1))
#            print(np.hstack([self.pop,outT]))
#            print(self.bestValue)
#            print("=============================================================")
        return self.pop[self.bestIndex].reshape((1,self.D))#返回最佳个体基因
    

#for i in range(5):       
#    F=0.5* (  2**exp(1- 5/(5-i)  ) )
#    print(F)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    