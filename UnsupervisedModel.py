# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:40:28 2017

@author: Zhdun
"""
class UnsupervisedModel:
    """
    简单无监督排序方法
    """
    def __init__(self,index=None,label=None):
        """
        index和label指向的是同一列，要么是索引index要么是标签label
        """
        self.index=index
        self.label=label
        
    def fit(self,X,y):
#        不做任何操作
        return
        
    def predict(self,X):
        if(self.index is not None):
            y=list(X.iloc[:,self.index])
        else:
            sign=self.label[-1]
            lab=self.label[0:-1]
            y=list(X.loc[:,lab])
            if(sign=='-'):    
                for i in range(len(y)):
                    y[i]=-1*y[i]
        return y





