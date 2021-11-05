#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def init(df, k):
    # return randomly selected k points with its range
    ix = np.random.choice(df.shape[0], size=k, replace=False)
    centroids = df.iloc[ix,:].reset_index(drop=True)
    return centroids

def euclidean(a, b):
    # return euclidean distance of two points
    return np.linalg.norm(a-b)

def assign(df, centroids):
    # calculate distances to centroids and assign to the closest centroid
    # return the clustering result in ndarray
    num_rows = df.shape[0]
    closest = np.zeros(num_rows)
    for i in range(num_rows):
        d = df.iloc[i,:]
        min_dist = float('inf')
        for centroid in range(centroids.shape[0]):
            c = centroids.iloc[centroid,:]
            dist = euclidean(d, c)
            if dist < min_dist:
                min_dist = dist
                closest[i] = centroid
    return closest


def update(df, clusters):
    # calculate new centroid as the mean of all points that belongs to the cluster
    # return updated centroids in dataframe
    return df.groupby(clusters).mean()


def k_means(dataset, k):
    centroids = init(dataset, k)
    clusters = []
    # assign points to clusters and update centroids
    # repeat until centroids no longer change
    while(True):
        clusters = assign(dataset, centroids)
        new_c = update(dataset, clusters)
        if new_c.equals(centroids):
            break
        else:
            centroids = new_c
    # format output
    clusters += 1
    clusters = clusters.astype(int)
    return clusters, centroids

# def read(file_path):

#     if file_path=='test.txt':
#         data = pd.read_csv(file_path,sep=r'\t', header=None, engine='python')
#         ds =  data.iloc[:,2:]
#         gt = data.iloc[:,1]

#     elif file_path=='adult.csv':
#         data = pd.read_csv(file_path,sep=',', header=None, engine='python')
#         data = data.apply(pd.to_numeric, errors='coerce')
#         ds =  data.iloc[1:,0:5]
#         gt = data.iloc[1:,5]
    
#     else: # spiral old or new
#         data = pd.read_csv(file_path,sep=',', header=None, engine='python')
#         data = data.apply(pd.to_numeric, errors='coerce')
#         ds =  data.iloc[1:,0:2]
#         gt = data.iloc[1:,2]
#     return ds, gt

# def visualization(data,labels,title,axis):
#     #draw the data points with a scatter plot, and color them according to their labels

#     labels_list = np.unique(labels)

#     fig, ax = plt.subplots()

#     for label in labels_list:
#         ix = np.where(np.array(labels) == label)
#         ax.scatter(data[ix,0],data[ix,1],label=label,s=15,alpha=0.5)
#     ax.legend()
#     plt.title(title)
#     plt.xlabel(axis + ' 1')
#     plt.ylabel(axis + ' 2')
#     plt.show()
#     return

# def visualize(dataset, labels, title):
#     # pca_res = mypca.pca(dataset.to_numpy())
#     visualization(dataset.to_numpy(), labels, title, 'PC')

# def main():

#     dataset1 =  'spiral_old.csv'
#     dataset2 = 'spiral.csv'
#     dataset3 = 'adult.csv'
#     dataset, ground_truth = read(dataset3)
#     # visualize(dataset, ground_truth, 'groundtruth')

#     # kmeans clustering
#     k_ = 10
#     kmeans_res, kmeans_centroids = k_means(dataset, k_)
#     # kmeans_ix = RandIndex(ground_truth, kmeans_res)
#     print('kmeans: k = {}'.format(k_))
#     # visualize(dataset, kmeans_res, 'kmeans')

# if __name__ == "__main__":
#     main()