# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:52:21 2017

@author: panpeng
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
            
# arr_x坐标的x值 arr_y坐标的y值 a,b,c对应L:ax+by+c=0
def mindistance( arr_x,arr_y,a,b,c ):
    if len(arr_x) != len(arr_y):
        return "list input error!"
    i = 0
    temp = abs(a*arr_x[0]+b*arr_y[0]+c)/((a*a+b*b)**0.5)
    for n in range(len(arr_x)):
        d = abs(a*arr_x[n]+b*arr_y[n]+c)/((a*a+b*b)**0.5)
        if d < temp:
            temp = d
            i = n
    return i

def Norder( data, order ):
    N = order
    df_id_train = data.copy()
    for i in range(len(df_id_train.columns)):
        data = df_id_train.iloc[:, i]
        for j in range(N):
            k = j
            value = data
            while (k > 0) :
                k = k-1
                value = value * data
            if (j > 0) :
                loc = len(df_id_train.columns)
                df_id_train.insert(loc, loc, value)
    return df_id_train
            
if __name__ == "__main__":
    ##    #nrows = 100
    # 读取数据
    #header=None,
    df_id_train = pd.read_csv("df_times_tag_train_percent_filled_sorted.csv", encoding='GBK', index_col = 0)
    
    df_test = pd.read_csv("df_test_final.csv", encoding='GBK', index_col = 0)
    

#    #删除标签
#
    a_data_0 = df_id_train[df_id_train['tag']==0] #提前0样本数据
    a_data_1 = df_id_train[df_id_train['tag'] == 1] #提前1样本数据
 
    del a_data_1["tag"]
    
    data_1_test, data_1 = train_test_split(a_data_1, train_size=0.1, random_state=1)
    
    non_data = data_1.copy()
    
    cls = KMeans(n_clusters=4, init='k-means++').fit(data_1.values)
    cc = cls.labels_ 
    data_1['tag'] = cc
          
    data_1_0 =  data_1[data_1['tag']==0] #提前0样本数据
    data_1_1 =  data_1[data_1['tag']==1] #提前0样本数据   
    data_1_2 =  data_1[data_1['tag']==2] #提前0样本数据
    data_1_3 =  data_1[data_1['tag']==3] #提前0样本数据 
    
    del data_1_0["tag"]
    del data_1_1["tag"]
    del data_1_2["tag"]
    del data_1_3["tag"]      
    
    NOC = 1200
    s = Smote(data_1_0.values, N=NOC)
    data_1_0_s = s.over_sampling()
    
    s_b=Smote(data_1_1.values, N=NOC)
    data_1_1_s = s_b.over_sampling()
    
    s=Smote(data_1_2.values, N=NOC)
    data_1_2_s = s.over_sampling()
    
    s=Smote(data_1_3.values, N=NOC)
    data_1_3_s = s.over_sampling()
    
    data_1_0_s = pd.DataFrame(data_1_0_s)
    data_1_1_s = pd.DataFrame(data_1_1_s)
    data_1_2_s = pd.DataFrame(data_1_2_s)
    data_1_3_s = pd.DataFrame(data_1_3_s)
    
    train_all_1 = pd.concat([data_1_0_s, data_1_1_s, data_1_2_s, data_1_3_s], axis=0)#1训练样本数据和标签整合
    train_all_1['tag'] = 1
    data_0_train, data_0_test = train_test_split(a_data_0, train_size=0.89, random_state=1)
    
    
    ####
    data_0_train =  pd.DataFrame(data_0_train.values)
    train_all_1 = pd.DataFrame(train_all_1.values)
    train_all = pd.concat([train_all_1, data_0_train], axis=0)#1训练样本数据和标签整合
    train_all = train_all.reset_index(drop=True)
    
    
    train_all_x = train_all.loc[:,0:31]
    train_all_y = train_all.loc[:,32]
    
    data_1_test['tag'] = 1
    data_1_test = data_1_test.reset_index(drop=True)
    data_0_test = data_0_test[0:1900]
    
    test_all = pd.concat([data_1_test, data_0_test  ], axis=0)#1训练样本数据和标签整合
    test_all = test_all.reset_index(drop=True)
    test_all = pd.DataFrame(test_all.values)
    test_all_x = test_all.loc[:,0:31]
    test_all_y = test_all.loc[:,32]

    N = 2
    train_all_x = Norder( train_all_x, N )
    test_all_x = Norder( test_all_x, N )


#####################
###生成pr数据
    non_data['tag'] = 1
    pr_1 = pd.DataFrame(non_data.values)  
    pr_0 = data_0_train[0:16000]
    
    pr = pd.concat([pr_1, pr_0], axis=0)#1训练样本数据和标签整合
    pr_x = pr.loc[:, 0:31]
    pr_y = pr.loc[:, 32]
    pr_x = Norder( pr_x, N )
    
    #数据标准化
    st = StandardScaler().fit(train_all_x)
    train_all_x = st.transform(train_all_x) 
    test_all_x = st.transform(test_all_x)
#    df_test = st.transform(df_test)
    
    ##############################
    #pr数据标准化
    pr_x = st.transform(pr_x) 
    
    
    #alpha的L2交叉验证
    alpha_can = np.logspace(-3, 2, 10)

    lr = LogisticRegressionCV(Cs = alpha_can, cv=5, scoring = 'f1', class_weight='balanced')

#    lr = LogisticRegression()
    #训练数据
    lr.fit(train_all_x, train_all_y.values.ravel())
#
#    ## 计算auc
#    yy = lr.predict_proba(train_all_x)
#    y1 = yy[:,1]
#    auc_train = metrics .roc_auc_score(train_all_y, y1)
#    
#    ### 计算roc
#    roc_train = metrics.roc_curve(train_all_y, y1)
#    plt.plot(roc_train[0], roc_train[1])
#    #################################################
#    
#    ## 计算auc
#    y_test = lr.predict_proba(test_all_x)
#    y1_test = y_test[:,1]
#    auc_test = metrics .roc_auc_score(test_all_y, y1_test)
#    
#    ### 计算roc    
#    roc_test = metrics.roc_curve(test_all_y, y1_test)
#    plt.plot(roc_test[0], roc_test[1])
    
#    ### 计算阈值
#    roc_threshold_loc = mindistance(roc_train[0], roc_train[1], 1, 1, -1)
#    roc_threshold = roc_train[2][roc_threshold_loc]
    
#    test_all_x = test_all_x[0:]

#####################################3
#计算PR平衡点
    PC = []
    RC = []
    FC = []
    yy = lr.predict_proba(pr_x)
    yy_1 = yy[:,1]
    for nn, j in enumerate(np.arange(0, 1, 0.01)):
        roc_threshold = j
        ###计算分类
        rat = roc_threshold
        y_hat = list(range(0, len(yy)))
        for i, j  in enumerate(yy_1):
            if yy_1[i] >= rat:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        y_hat = np.array(y_hat, dtype = float)
        
        #计算准确度
        y = pr_y.values.reshape(-1)
#        result = y_hat == y
#        print (y_hat)
#        print (result)
#        acc = np.mean(result)
#        print ('准确度: %.2f%%' % (100 * acc))
    
        PR= precision_recall_fscore_support(y, y_hat, average='binary') 
        PC.append(PR[0])
        RC.append(PR[1])
        FC.append(PR[2])
        
    pr_threshold_loc = mindistance(PC, RC, 1, -1, 0)
    pr_threshold = pr_threshold_loc/100
        
    plt.plot(PC, RC)    
    
###########################    
    rat = pr_threshold
    yy = lr.predict_proba(pr_x)
    yy_1 = yy[:,1]
    y_hat = list(range(0, len(yy)))
    for i, j  in enumerate(yy_1):
        if yy_1[i] >= rat:
            y_hat[i] = 1
        else:
            y_hat[i] = 0
    y_hat = np.array(y_hat, dtype = float)
    y = pr_y.values.reshape(-1)
    PR= precision_recall_fscore_support(y, y_hat, average='binary') 
    print("精确度:",PR[0])
    print("召回率:",PR[1])
    print("F1值:",PR[2])
        
        

