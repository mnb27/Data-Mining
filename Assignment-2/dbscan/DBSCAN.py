# DBSCAN Clustering

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
# dataset = pd.read_csv('spiral_new.csv')
# X = dataset.iloc[:, [0, 1]].values
# X = X.to_numpy()

dataset = pd.read_csv('adult.csv')
X = dataset.iloc[:, [0, 4]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.65,min_samples=50)

# Fitting the model

model=dbscan.fit(X)

labels=model.labels_


from sklearn import metrics

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True
# print("SAMPLE CORES",sample_cores[:50])

# print("LABELS",labels[:50])

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)
print("Clusters",n_clusters)


print(metrics.silhouette_score(X,labels))



