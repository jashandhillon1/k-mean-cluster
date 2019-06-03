# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:03:25 2019

@author: jashanpreet singh
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]]

#using the dandogram to find the no of clusters
import scipy.cluster.hierarchy as sch
dandogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('dandogram')
plt.xlabel('customers')
plt.ylabel('distances')
plt.show()

#fitting the model
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y=hc.fit_predict(x)