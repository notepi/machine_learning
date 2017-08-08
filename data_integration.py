# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:57:09 2017

@author: panpeng
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
      #header=None,
#    df_id_train = pd.read_csv("df_test.csv")
    df_id_train = pd.read_csv("df_train.csv")
    #统计次数
    times_count = df_id_train["个人编码"].value_counts()
    times_count = pd.Series.to_frame(times_count)
    #修改列名称
    times_count.rename(columns={'个人编码': 'times'}, inplace=True) 
    
    ### 去掉某些特征值
    del df_id_train["交易时间"]
    del df_id_train["住院开始时间"]
    del df_id_train["操作时间"]
    del df_id_train["住院终止时间"]    
    del df_id_train["申报受理时间"]
    del df_id_train["出院诊断病种名称"]
    del df_id_train["顺序号"]
    del df_id_train["医院编码"]
    del df_id_train["农民工医疗救助计算金额"]
    del df_id_train["最高限额以上金额"]
    del df_id_train["双笔退费标识"]
    del df_id_train["住院天数"]
    del df_id_train["家床起付线剩余"]
    del df_id_train["手术费拒付金额"]
    del df_id_train["手术费申报金额"]
    del df_id_train["床位费拒付金额"]
    del df_id_train["床位费申报金额"]
    del df_id_train["残疾军人医疗补助基金支付金额"]
    del df_id_train["医用材料费拒付金额"]
    del df_id_train["输全血申报金额"]
    del df_id_train["成分输血自费金额"]
    del df_id_train["成分输血拒付金额"]
    del df_id_train["其它拒付金额"]
    del df_id_train["一次性医用材料自费金额"]
    del df_id_train["一次性医用材料拒付金额"]
    del df_id_train["输全血按比例自负金额"]
    del df_id_train["统筹拒付金额"]
    del df_id_train["非典补助补助金额"]
    del df_id_train["检查费拒付金额"]
    del df_id_train["药品费拒付金额"]
    del df_id_train["治疗费拒付金额"]
    
    df_id_train = df_id_train.groupby(['个人编码']).sum()
    df_id_train = round(df_id_train, 2) # reduce the precision to 2 deci
    
    #填充数据
    df_id_train = df_id_train.fillna(0)
    #合并数据，添加times参数
    df_id_train = pd.merge(times_count, df_id_train, left_index=True, right_index=True, how='outer')
    
    ###生成测试数据
    hh = df_id_train[u'贵重药品发生金额'] / df_id_train[u'药品费发生金额']
    del df_id_train["贵重药品发生金额"]    
    df_id_train.insert(3,'贵重药费比例',hh)
    #填充数据
    df_id_train.fillna({'贵重药费比例': 0}, inplace = True)   
    
    hh = df_id_train[u'中成药费发生金额'] / df_id_train[u'药品费发生金额']
    del df_id_train["中成药费发生金额"]    
    df_id_train.insert(4,'中成药费比例',hh)
    #填充数据
    df_id_train.fillna({'中成药费比例': 0}, inplace = True)   
    
    hh = df_id_train[u'中草药费发生金额'] / df_id_train[u'药品费发生金额']
    del df_id_train["中草药费发生金额"]    
    df_id_train.insert(5,'中草药费比例',hh) 
    #填充数据
    df_id_train.fillna({'中草药费比例': 0}, inplace = True)   
    
    hh = df_id_train[u'药品费自费金额'] / df_id_train[u'药品费发生金额']
    del df_id_train["药品费自费金额"]    
    df_id_train.insert(6,'药品费自费比例',hh)
    #填充数据
    df_id_train.fillna({'药品费自费比例': 0}, inplace = True)      
    
    hh = df_id_train[u'药品费申报金额'] / df_id_train[u'药品费发生金额']
    del df_id_train["药品费申报金额"]    
    df_id_train.insert(7,'药品费申报金额比例',hh)   
    #填充数据
    df_id_train.fillna({'药品费申报金额比例': 0}, inplace = True)    
    
### 检测部分
    
    hh = df_id_train[u'贵重检查费金额'] / df_id_train[u'检查费发生金额']
    del df_id_train["贵重检查费金额"]    
    df_id_train.insert(9,'贵重检查费比例',hh)    
    #填充数据
    df_id_train.fillna({'贵重检查费比例': 0}, inplace = True)    
    
    hh = df_id_train[u'检查费自费金额'] / df_id_train[u'检查费发生金额']
    del df_id_train["检查费自费金额"]    
    df_id_train.insert(10,'检查费自费比例',hh)    
    #填充数据
    df_id_train.fillna({'检查费自费比例': 0}, inplace = True)    
    
    hh = df_id_train[u'检查费申报金额'] / df_id_train[u'检查费发生金额']
    del df_id_train["检查费申报金额"]    
    df_id_train.insert(11,'检查费申报比例',hh)
    #填充数据
    df_id_train.fillna({'检查费申报比例': 0}, inplace = True)    
    
### 治疗部分

    hh = df_id_train[u'治疗费自费金额'] / df_id_train[u'治疗费发生金额']
    del df_id_train["治疗费自费金额"]    
    df_id_train.insert(13,'治疗费自费比例',hh) 
    #填充数据
    df_id_train.fillna({'治疗费自费比例': 0}, inplace = True)
    
    hh = df_id_train[u'治疗费申报金额'] / df_id_train[u'治疗费发生金额']
    del df_id_train["治疗费申报金额"]    
    df_id_train.insert(14,'治疗费申报比例',hh) 
    #填充数据
    df_id_train.fillna({'治疗费申报比例': 0}, inplace = True)
    
### 手术部分

    hh = df_id_train[u'手术费自费金额'] / df_id_train[u'手术费发生金额']
    del df_id_train["手术费自费金额"]    
    df_id_train.insert(16,'手术费自费比例',hh)
    #填充数据
    df_id_train.fillna({'手术费自费比例': 0}, inplace = True)
    
### 医用材料发生金额    

    hh = df_id_train[u'高价材料发生金额'] / df_id_train[u'医用材料发生金额']
    del df_id_train["高价材料发生金额"]    
    df_id_train.insert(18,'高价材料金额比例',hh)
    #填充数据
    df_id_train.fillna({'高价材料金额比例': 0}, inplace = True)

    hh = df_id_train[u'医用材料费自费金额'] / df_id_train[u'医用材料发生金额']
    del df_id_train["医用材料费自费金额"]    
    df_id_train.insert(19,'医用材料费自费金额比例',hh)
    #填充数据
    df_id_train.fillna({'医用材料费自费金额比例': 0}, inplace = True)
    
### 其他
    hh = df_id_train[u'其它申报金额'] / df_id_train[u'其它发生金额']
    del df_id_train["其它申报金额"]    
    df_id_train.insert(23,'其它申报金额比例',hh)
    #填充数据
    df_id_train.fillna({'其它申报金额比例': 0}, inplace = True)
    
    
###小计费用
    hh =    df_id_train[u'药品费发生金额'] + \
            df_id_train[u'检查费发生金额'] + \
            df_id_train[u'治疗费发生金额'] + \
            df_id_train[u'手术费发生金额'] + \
            df_id_train[u'床位费发生金额'] + \
            df_id_train[u'医用材料发生金额'] + \
            df_id_train[u'其它发生金额'] + \
            df_id_train[u'一次性医用材料申报金额']
    df_id_train.insert(1,'小计费用',hh)
    
    hh = df_id_train[u'药品费发生金额'] / df_id_train[u'小计费用']
    del df_id_train["药品费发生金额"]
    df_id_train.insert(2,'药品费比例',hh)
    #填充数据
    df_id_train.fillna({'药品费比例': 0}, inplace = True)

    hh = df_id_train[u'检查费发生金额'] / df_id_train[u'小计费用']
    del df_id_train["检查费发生金额"]
    df_id_train.insert(8,'检查费比例',hh)
    #填充数据
    df_id_train.fillna({'检查费比例': 0}, inplace = True) 

    hh = df_id_train[u'治疗费发生金额'] / df_id_train[u'小计费用']
    del df_id_train["治疗费发生金额"]    
    df_id_train.insert(12,'治疗费比例',hh) 
    #填充数据
    df_id_train.fillna({'治疗费比例': 0}, inplace = True) 

    hh = df_id_train[u'手术费发生金额'] / df_id_train[u'小计费用']
    del df_id_train["手术费发生金额"]     
    df_id_train.insert(15,'手术费比例',hh)
    #填充数据
    df_id_train.fillna({'手术费比例': 0}, inplace = True)          

    hh = df_id_train[u'床位费发生金额'] / df_id_train[u'小计费用']
    del df_id_train["床位费发生金额"]      
    df_id_train.insert(17,'床位费比例',hh)   
    #填充数据
    df_id_train.fillna({'床位费比例': 0}, inplace = True)      

    hh = df_id_train[u'医用材料发生金额'] / df_id_train[u'小计费用']
    del df_id_train["医用材料发生金额"]      
    df_id_train.insert(18,'医用材料费比例',hh)
    #填充数据
    df_id_train.fillna({'医用材料费比例': 0}, inplace = True) 
                      

    hh = df_id_train[u'其它发生金额'] / df_id_train[u'小计费用']
    del df_id_train["其它发生金额"] 
    df_id_train.insert(22,'其他费比例',hh)
    #填充数据
    df_id_train.fillna({'其他费比例': 0}, inplace = True)     

    hh = df_id_train[u'一次性医用材料申报金额'] / df_id_train[u'小计费用']
    del df_id_train["一次性医用材料申报金额"] 
    df_id_train.insert(23,'一次性医用材料费比例',hh)
    #填充数据
    df_id_train.fillna({'一次性医用材料费比例': 0}, inplace = True)                                                                           
        

#    df_id_train.to_csv('df_test_final.csv',encoding = "GBK")
    df_id_train.to_csv('df_train_final.csv',encoding = "GBK")
    
    
    
    
    
    
    
    
    
    
    