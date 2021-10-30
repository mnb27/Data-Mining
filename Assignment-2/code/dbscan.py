#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pca as mypca

def DBSCAN(data, eps, minpts):
    data = data.to_numpy()
    cluster = 0
    visited = np.zeros(len(data))
    labels = np.zeros(len(data))
    
    for i in range(len(data)):
        if visited[i]==0:
            visited[i]=1
            neighborPts = regionQuery(data, i, eps)
            if len(neighborPts)<minpts:
                labels[i]=-1
            else:
                #expandCluster
                cluster += 1
                labels[i]=cluster
                j=0
                while j<len(neighborPts):
                    index = neighborPts[j]
                    if visited[index]==0:
                        visited[index]=1
                        new_neighbor = regionQuery(data, index, eps)
                        if len(new_neighbor)>=minpts:
                            neighborPts = neighborPts+new_neighbor
                    if labels[index]==0:
                        labels[index]=cluster
                    j+=1
            
                
    return labels


def regionQuery(data, center, eps):
    neighborPts = []
    for i in range(len(data)):
            dist = np.linalg.norm(data[center]-data[i])
            if(dist<eps):
                neighborPts.append(i)
    return neighborPts

def distance(a,b):
    res = 0
    for i in range(len(a)):
        res += (a[i]-b[i])*(a[i]-b[i])
    res = math.sqrt(res)
    return res

def read(file_path):

    if file_path=='test.txt':
        data = pd.read_csv(file_path,sep=r'\t', header=None, engine='python')
        ds =  data.iloc[:,2:]
        gt = data.iloc[:,1]

    elif file_path=='adult.csv':
        data = pd.read_csv(file_path,sep=',', header=None, engine='python')
        data = data.apply(pd.to_numeric, errors='coerce')
        ds =  data.iloc[1:,0:5]
        gt = data.iloc[1:,5]
    
    else: # spiral old or new
        data = pd.read_csv(file_path,sep=',', header=None, engine='python')
        data = data.apply(pd.to_numeric, errors='coerce')
        ds =  data.iloc[1:,0:2]
        gt = data.iloc[1:,2]
    return ds, gt

def visualization(data,labels,title,axis):
    #draw the data points with a scatter plot, and color them according to their labels

    labels_list = np.unique(labels)

    fig, ax = plt.subplots()

    for label in labels_list:
        ix = np.where(np.array(labels) == label)
        ax.scatter(data[ix,0],data[ix,1],label=label,s=15,alpha=0.5)
    ax.legend()
    plt.title(title)
    plt.xlabel(axis + ' 1')
    plt.ylabel(axis + ' 2')
    plt.show()
    return

def visualize(dataset, labels, title):
    # pca_res = mypca.pca(dataset.to_numpy())
    visualization(dataset.to_numpy(), labels, title, 'PC')
    # visualization(pca_res, labels, title, 'PC')

def main():
    dataset1 =  'spiral_old.csv'
    dataset2 = 'spiral.csv'
    dataset3 = 'adult.csv'
    dataset, ground_truth = read(dataset2)
    visualize(dataset, ground_truth, 'groundtruth')

    # DBSCAN
    for eps in [0.4]:
         for min_samples in [10]:
             DBSCAN_res = DBSCAN(dataset, eps, min_samples)
             unique_labels = set()
             for label in DBSCAN_res:
                 unique_labels.add(label)
             print('DBSCAN: eps = {}, minPts = {}'.format(eps, min_samples))
             visualize(dataset, DBSCAN_res, 'DBSCAN')
             print("Clusters: ",unique_labels)
             print("Actual: ",ground_truth[:5])
             print("Predicted: ",DBSCAN_res[:5])

if __name__ == "__main__":
    main()