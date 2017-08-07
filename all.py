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
##    #nrows = 100
#    df_id_train = pd.read_csv("df_train_count_filled_sorted.csv", encoding = "GBK")
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
#  表整合
    times_count = pd.read_csv("df_train_count_filled_sorted.csv", encoding = "GBK")
    df_id_train = pd.read_csv("df_id_train.csv", encoding = "GBK",names=["key", "tags"])
    merge_data = pd.merge(times_count, df_id_train, how='left', on=None,
                          left_on="个人编码", right_on="key", copy=False, indicator=False)  
    print(merge_data)
#    df_id_train.to_csv('df_train_count_filled_sorted.csv',encoding = "GBK")
    



















