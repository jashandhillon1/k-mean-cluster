# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:44:16 2019

@author: jashanpreet singh
"""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#using the elbow method to get the no. of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('THE ELBOW METHOD')
plt.xlabel('clusters')
plt.ylabel('wcss')
plt.show()

#fitting the model
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,random_state=0)
y=kmeans.fit_predict(x)

#visualizing the result
plt.scatter(x[y==0,0],x[y==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y==1,0],x[y==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[y==2,0],x[y==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y==3,0],x[y==3,1],s=100,c='yellow',label='cluster4')
plt.scatter(x[y==4,0],x[y==4,1],s=100,c='cyan',label='cluster5')
plt.show()