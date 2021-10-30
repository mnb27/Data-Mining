from os import path
import numpy as np
import random
import math
from datetime import datetime
# import matplotlib.pyplot as plt
import pandas as pd

class KMeans_class:
	def __init__(self, data):
		self.data = data[:, 0:data.shape[1]] # n*D
		self.labels = data[:, data.shape[1] - 1] # n*1
		self.K = 2
		self.cluster_label = {}

	def random_initialiser(self):
		# K : no. of clusters
		# for centroids we can pick K randm data points w/o loss of generality
		temp = random.sample(range(0, self.data.shape[0]), self.K)
		centroids = np.zeros((self.K, self.data.shape[1])) # K * D
		z = 0
		for i in temp:
			centroids[z] = self.data[i]
			z += 1
		return centroids
		

	def cluster_centroid(self, points_in_cluster):
		return np.mean(points_in_cluster, axis=0)


	def dist(self, x, y):
		sum_ = 0
		for d in range(len(x)):
			sum_ += (x[d] - y[d]) ** 2
		return math.sqrt(sum_)


	def calculate_sse(self, points_in_cluster):
		sse = 0
		centroids = []
		for i in range(self.K):
			centroids.append(self.cluster_centroid(points_in_cluster[i]))
		centroids = np.array(centroids) # K * D
		for cluster_no in points_in_cluster.keys():
			points = points_in_cluster[cluster_no]
			points = np.array(points) # N_j * D
			sum_x_T_x = 0
			for point in points:
				# print("Debug --- ",point,"---",centroids[cluster_no])
				sum_x_T_x += np.linalg.norm(point[:-1] - centroids[cluster_no][:-1]) # 1*1
			sse += sum_x_T_x
		return sse


	def find_closest_centroid(self, distance_from_centroid):
		points_in_cluster = {} # len = K
		assigned_clusters = [] # len = n
		for dist_list in distance_from_centroid:
			# index of minimum value in list
			assigned_clusters.append( dist_list.index(min(dist_list)) )
		z = 0
		for i in range( self.K ):
			points_in_cluster[i] = []
		for i in assigned_clusters:
			# points_in_cluster[i].append(self.dataa[z]) # data[z] is not list, it's 1-D array
			points_in_cluster[i].append(self.data[z]) # data[z] is not list, it's 1-D array
			z += 1
		#print(assigned_clusters, points_in_cluster)
		return assigned_clusters, points_in_cluster


	def K_Means_util(self, centroids):
		previous_centroids = centroids

		# calculate distance of each points from centroids
		distance_from_centroid = [] # len = n, each element in this list is a list of k length
		for i in self.data[:,0:5]:
			dis = []
			for centroid in centroids:
				# print(i,"---",centroid[0:5])
				dis.append(self.dist(i, centroid[0:5]))
			distance_from_centroid.append(dis)

		# assign each point to the cluster which has minimum distance between point and cluster centroid
		assigned_clusters, points_in_cluster = self.find_closest_centroid(distance_from_centroid)

		# update cluster mean/centroid
		new_centroids = np.zeros((self.K, self.data.shape[1]))
		for cluster_no in points_in_cluster.keys():
			new_centroids[cluster_no] = self.cluster_centroid(points_in_cluster[cluster_no])

		# if new centroid is same as previous centroid then return else repeat
		if np.array_equal(previous_centroids, new_centroids):
			return new_centroids, assigned_clusters, points_in_cluster
		else:
			return self.K_Means_util(new_centroids)


	def K_Means_unlabelled(self):
		# data : (n*d) array
		start = datetime.now()
		centroids = self.random_initialiser()
		best_centroid, best_assigned_clusters, best_points_in_cluster = self.K_Means_util(centroids)
		end = datetime.now()
		time_taken = round((end-start).total_seconds(), 2)
		return best_centroid, best_assigned_clusters, best_points_in_cluster, time_taken

	def run_on_K(self, K):
		self.K = K
		centroid, assigned_clusters, points_in_cluster, time = self.K_Means_unlabelled()
		return centroid, assigned_clusters, points_in_cluster, time, list(self.labels)

def main():
    df = pd.read_csv('adult.csv')
    data = df.to_numpy()
    # print("DATA: ", data)
    # print()
    # print(data[:, 0:data.shape[1]])
    # print(data[:, 0:data.shape[1]][:,0:5])
    # print()
    # print(data[:, data.shape[1] - 1])

    obj = KMeans_class(data)
    res = obj.run_on_K(10)
    
    # print("Centroid: ",res[0])
    # print()
    # print("Assigned Clusters: ",res[1])
    # print()
    # print("Points in cluster: ",res[2])
    # print()
    # print(res[4])
    print("SSE: ",obj.calculate_sse(res[2]))
    label = res[1]
    gender = res[4]

    #### GROUP FAIRNESS NOTION
    freqMale = {}
    freqFemale = {}
    for item in set(label):
        freqMale[item] = 0
        freqFemale[item] = 0

    for i in range(len(label)):
        item = label[i]
        if gender[i]==1.0:
            if (item in freqMale):
                freqMale[item] += 1
            else:
                freqMale[item] = 1
        else :
            if (item in freqFemale):
                freqFemale[item] += 1
            else:
                freqFemale[item] = 1

    ratioInEachCluster = list()
    for id in list(set(label)):
        a = freqMale[id]
        b = freqFemale[id]
        if(b!=0): ratioInEachCluster.append(a/b)
        else: ratioInEachCluster.append(float('inf'))
        # print(id," --- ",freqFemale[id]," --- ",freqMale[id])
        # print()
    print("Ratio array: ",ratioInEachCluster)
    balance = min(ratioInEachCluster)
    print("Balance: ",balance)
	
    ### Individual fairness notion
    clusterCentroidsWithLabel = res[0]
    pointsInClusterwithLabel = res[2]
    clusterCentroids = clusterCentroidsWithLabel[:,0:5]
    
    p = []
    for id in range(5):
        fartestDistMalePoint = float('-inf')
        fartestDistFemalePoint = float('-inf')
		# print(id," --- ", pointsInCluster[id])
        for point in pointsInClusterwithLabel[id]:
            gender = point[5]
            distance = obj.dist(clusterCentroids[id], point[:-1])
            if gender==1.0 and distance > fartestDistMalePoint:
                fartestDistMalePoint = distance
            elif gender==0.0 and distance > fartestDistFemalePoint:
                fartestDistFemalePoint = distance    
        if fartestDistFemalePoint!=0.0: p.append(fartestDistMalePoint/fartestDistFemalePoint)
        else: p.append(float('inf'))
    print("P array: ",p)
    maxD = min(p)
    print("MaxD: ",maxD)
    # print(clusterCentroids[:,0:5])
    # print(pointsInClusterwithLabel[0][1][5])  

if __name__ == "__main__":
    main()