# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:38:52 2017

@author: panpeng
"""

import csv
import numpy as np
import pandas as pd
from decimal import *

###
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
    df_id_train = pd.read_csv("df_times_tag_train_percent_filled_sorted.csv", encoding='GBK', nrows = 100)
    
    df_id_train = Norder(df_id_train , 1)













