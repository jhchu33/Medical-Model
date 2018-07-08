# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:02:38 2017

@author: Administrator
"""



import pandas as pd
import jieba
from sklearn.externals import joblib
from keras.preprocessing import sequence
import numpy as np



model = joblib.load("D:\jk_zb_yongyao_dec_tree.m")
p =model.predict([[0,1,3,0,0,1,1,1]])
print(p)