# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:09:47 2017

@author: Zhdun
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
class SMOTER:
    """
    实现SMOTER的功能
    """
    def __init__(self,k,m,r,data):
        self.k=k
        self.m=m
        self.r=r
        self.numInstances=None
        self.data=data

    def __KNearestNeighbor(self,data,target):
        """
    
        Parameters:
        ---------------------
            data:
                ndarray类型，包含X和y
            
            target:
                行向量，包含X和y
            
            k:
              k个最近的
        Return:
        -------
        
        data中与target k个最近的记录的索引
        """
        distances = np.zeros((data.shape[0],1))
        numAttrs=data.shape[1]
        resIndex=[]
        for i in range(data.shape[0]):
#            distances[i,0]=np.linalg.norm(data[i].reshape(1,numAttrs)[0,0:numAttrs-1]-target[0,0:numAttrs-1])
            distances[i,0]=pdist(np.vstack([data[i].reshape(1,numAttrs)[0,0:numAttrs-1],target[0,0:numAttrs-1]]),p=self.r,metric='minkowski')[0]
        for i in range(int(self.k)):
            resIndex.append(distances.argmin())
            distances[int(resIndex[-1]),0]=float("inf")
        return resIndex
    def __createASample(self,source,target):
        """
        Parameters:
        ------------
        source  target:
                两个行向量,包含X和y
        
        Return:
        --------
        根据source和target向量生成新的行向量
        """
        s,t=np.array(source),np.array(target)
        sy,ty=s[0,-1],t[0,-1]
        s[0,-1],t[0,-1]=0,0
        res=s+(t-s)*np.random.rand(1,s.shape[1])
        
#        d1,d2=np.linalg.norm(res-t),np.linalg.norm(res-s)
        d1=pdist(np.vstack([res,t]),p=self.r,metric='minkowski')[0]
        d2=pdist(np.vstack([res,s]),p=self.r,metric='minkowski')[0]
        if((d1+d2)==0):
            res[0,-1]=sy
        else:
            res[0,-1]=(d1*sy+d2*ty)/(d1+d2)
        return res
    
    def smoteR(self):
        #获取少数类的样本集合
        d=np.array(self.data)
        
        t=[]
        for i in range(d.shape[0]):
            if(d[i,-1]!=0):
                t.append(i)
        rareData=d[t,:]
#        print(len(d),len(rareData))
        #确定每个少数类应该生成多少实例
        self.numInstances=int((len(d)-2*len(rareData))/6.0*self.m)
        count=[int(self.numInstances/len(rareData))]*int(len(rareData))
        for i  in range( int(self.numInstances) % int(len(rareData))  ):
            count[i]=count[i]+1
        #开始生成实例
        for i in range(rareData.shape[0]):#对于每一个少数类样本
            target=rareData[i].reshape((1,d.shape[1]))
                    
            t_rareData=np.delete(rareData,i,0)#不包含目标少数类样本的的少数类样本集
    
            kNearest=self.__KNearestNeighbor(t_rareData,target)#找与目标实例最近的k个实例
            for j in range(int(count[i])):#为这个少数类样本生成相应数量的样本
                randomIndex= kNearest[np.random.randint(len(kNearest))]
                synthetic=self.__createASample(t_rareData[randomIndex].reshape(1,d.shape[1]),target)
                d=np.append(d,synthetic,axis=0)
        self.data= pd.DataFrame(d,columns=self.data.columns)
        ############################
#        td=deepcopy(self.data)
#        t1=len(td[td["bug"]==0])
#        t2=len(td[td["bug"]!=0])
#        print(self.numInstances,t1,t2)
        ############################
        return self.data






