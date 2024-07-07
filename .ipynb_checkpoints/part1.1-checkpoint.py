# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:17:25 2024

@author: garviagrawal
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot')
# pd.set_option ('max_column',200)

df = pd.read_csv('C:\\Users\\garviagrawal\\Downloads\\Policyfeatures.csv')
# print(df.head())
# print(df.shape)
print(df.describe)