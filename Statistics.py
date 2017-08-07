# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:38:52 2017

@author: panpeng
"""

import csv
import numpy as np
import pandas as pd
from decimal import *

if __name__ == "__main__":
##   统计每类标签的数目    
#    path = 'df_id_train.csv'
#    data = pd.read_csv(path, names=["key", "tags"])
##    统计某一列x中各个值出现的次数：a['x'].value_counts()
#    tags_count = data["tags"].value_counts()
#    print(type(tags_count))
#    tags_count.to_csv('tags_count.csv', index_label = ["tags", "counts"])  

##  统计每个人报销的次数
#df.to_csv(‘/tmp/9.csv’,columns=[‘open’,’high’],index=False,header=False)
#不要列头，不要索引，只要open,high两列。
#   data.to_csv('test.csv',index=False ) 

# nrows = 100,行数
#    data = pd.read_csv("df_train_sorted.csv", encoding = "GBK")
#    
#    tags_count = data["个人编码"].value_counts()
#    tags_count.to_csv('times_count.csv', headers =True, 
#                      index_label = ["tags", "counts"]) 
#    print(tags_count)

##  表整合
#    times_count = pd.read_csv("times_count.csv", encoding = "GBK",names=["id", "times"])
#    df_id_train = pd.read_csv("df_id_train.csv", encoding = "GBK",names=["key", "tags"])
##    print(df_id_train)
#    merge_data = pd.merge(times_count, df_id_train, how='left', on=None,
#                          left_on="id", right_on="key", copy=False, indicator=False)  
#    del merge_data["key"]
#    print(merge_data)   
#    merge_data.to_csv('tags_counts_merge_data.csv', index = False) 

### 查询缺失数据
#    data_merge_drug = pd.read_csv("data_merge_drug.csv", encoding = "GBK")
#    null_num = data_merge_drug[data_merge_drug['tag'].isnull()]
#    print(null_num[u'个人编码'].unique())

###数据求和
#    #nrows = 100
#    data_merge_drug = pd.read_csv("data_merge_drug.csv", encoding = "GBK")
#    gd = data_merge_drug.groupby(['个人编码']).sum()
#    #保留两位小数
#    gd = round(gd,2) # reduce the precision to 2 deci
##    gd.to_csv('data_merge_drug_count.csv',encoding = "utf-8")
##统计数据
##    gd_one = gd[gd['tag']>0.1]
##    gd_one["药品费发生金额"].hist(bins=500)
#    gd_zero = gd[gd['tag']<0.1]
#    gd_zero["药品费发生金额"].hist(bins=1000)
##    print(gd_one)
##    ## 查询缺失数据
##    print(df_id_train)
#    del df_id_train["交易时间"]
#    del df_id_train["住院开始时间"]
#    del df_id_train["操作时间"]
#    del df_id_train["住院终止时间"]    
#    del df_id_train["申报受理时间"]
#    del df_id_train["出院诊断病种名称"]
#    del df_id_train["顺序号"]
#    del df_id_train["医院编码"]
#    df_id_train.to_csv('demo.csv', index = False)

#    df_id_train = df_id_train.groupby(['个人编码']).sum()
###    #保留两位小数
#    df_id_train = round(gd,2) # reduce the precision to 2 deci
#    print(df_id_train)
#    del df_id_train["农民工医疗救助计算金额"]
#    del df_id_train["最高限额以上金额"]
#    del df_id_train["双笔退费标识"]
#    del df_id_train["住院天数"]
#    del df_id_train["家床起付线剩余"]
#    del df_id_train["手术费拒付金额"]
#    del df_id_train["手术费申报金额"]
#    del df_id_train["床位费拒付金额"]
#    del df_id_train["床位费申报金额"]
#    del df_id_train["残疾军人医疗补助基金支付金额"]
#    del df_id_train["医用材料费拒付金额"]
#    del df_id_train["输全血申报金额"]
#    del df_id_train["成分输血自费金额"]
#    del df_id_train["成分输血拒付金额"]
#    del df_id_train["其它拒付金额"]
#    del df_id_train["一次性医用材料自费金额"]
#    del df_id_train["一次性医用材料拒付金额"]
#    del df_id_train["输全血按比例自负金额"]
#    del df_id_train["统筹拒付金额"]
#    del df_id_train["非典补助补助金额"]
#    del df_id_train["检查费拒付金额"]
#    del df_id_train["药品费拒付金额"]
#    del df_id_train["治疗费拒付金额"]


###   填补数据
#    df_id_train = df_id_train.fillna(0)
#    print(df_id_train)

#    gd_zero = df_id_train[df_id_train['住院天数']>0.1]
#    print(gd_zero[u'住院天数'])
##  表整合
#    times_count = pd.read_csv("df_train_count_filled_sorted.csv", encoding = "GBK")
#    df_id_train = pd.read_csv("df_id_train.csv", encoding = "GBK",names=["个人编码", "tag"])
#    merge_data = pd.merge(times_count, df_id_train, how='left', on=None,
#                          left_on="个人编码", right_on="个人编码", copy=False, indicator=False)

##  表整合
#    df_id_train = pd.read_csv("df_tag_train_count_filled_sorted.csv", encoding = "GBK")
#    times_count = pd.read_csv("times_count_merge_data.csv", encoding = "GBK")
#    del times_count["tag"]
#    print(df_id_train)
#    merge_data = pd.merge(times_count, df_id_train, how='left', on=None,
#                          left_on="个人编码", right_on="个人编码", copy=False, indicator=False)
#    print(merge_data)
#    merge_data.to_csv('df_times_tag_train_count_filled_sorted.csv',encoding = "GBK", index = False)
#    




















