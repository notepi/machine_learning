# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:57:09 2017

@author: panpeng
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    path = u'8.iris.data'  # 数据文件路径
    # 读取数据 路径，浮点型数据，逗号分隔，
    data = np.loadtxt(path, dtype=float, delimiter=',')
    # 将数据的0到3列组成x，第4列得到y
    x, y = np.split(data, (4,), axis=1)
    # 使用前3列特征
    x = x[:, :3]
    

    ## 训练集上的预测结果
    
    #数据标准化
    x = StandardScaler().fit_transform(x)
    
    #alpha的L2交叉验证
    alpha_can = np.logspace(-3, 2, 10)
    lr = LogisticRegressionCV(Cs = alpha_can, cv=5)
    #训练数据
    lr.fit(x,y.ravel())
    
    # 预测   
    y_hat = lr.predict(x)
    
    #读取模型系数
    print("value is:")
    print(lr.coef_)
    print("==========================")
    #计算概率
    y_poss = lr.predict_proba(x)
    print(y_poss)
    
    #
    lr.get_params()
    
    #计算准确度
    y = y.reshape(-1)
    result = y_hat == y
    print (y_hat)
    print (result)
    acc = np.mean(result)
    print ('准确度: %.2f%%' % (100 * acc))

