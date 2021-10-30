import numpy as np
import random
import math
from datetime import datetime
import matplotlib.pyplot as plt


class KMeans_class:
	def __init__(self, data):
		self.data = data[:, 0:data.shape[1] - 1] # n*D
		self.labels = data[:, data.shape[1] - 1] # n*1
		self.K = 2
		self.cluster_label = {}


	def random_initialiser(self):
		# K : no. of clusters
		# for centroids we can pick K randm data points w/o loss of generality
		temp = random.sample(range(0, self.data.shape[0]-1), self.K)
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
				sum_x_T_x += np.linalg.norm(point - centroids[cluster_no]) # 1*1
			sse += sum_x_T_x
		return sse


	def classification_accuracy(self, assigned_clusters):
		cluster_labels_class_wise = {} # len = K, each element is a dictionary containing label:support
		for i in range(self.data.shape[0]):
			label = self.labels[i]
			assigned_cluster = assigned_clusters[i]
			if assigned_cluster not in cluster_labels_class_wise.keys():
				cluster_labels_class_wise[assigned_cluster] = {}
				cluster_labels_class_wise[assigned_cluster][label] = 1
			else:
				if label not in cluster_labels_class_wise[assigned_cluster].keys():
					cluster_labels_class_wise[assigned_cluster][label] = 1
				else:
					cluster_labels_class_wise[assigned_cluster][label] += 1
		#print(cluster_labels_class_wise)
		for cluster_no in cluster_labels_class_wise.keys():
			majority = 0
			for label in cluster_labels_class_wise[cluster_no].keys():
				if majority <= cluster_labels_class_wise[cluster_no][label]:
					self.cluster_label[cluster_no] = label
					majority = cluster_labels_class_wise[cluster_no][label]
		misclassified = 0
		for i in range(self.data.shape[0]):
			label = self.labels[i]
			assigned_cluster = assigned_clusters[i]
			cluster_class = self.cluster_label[assigned_cluster]
			if not (label == cluster_class):
				misclassified += 1
		return round( ( (self.data.shape[0]-misclassified)*100 / (self.data.shape[0])), 2 )



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
			points_in_cluster[i].append(self.data[z]) # data[z] is not list, it's 1-D array
			z += 1
		#print(assigned_clusters, points_in_cluster)
		return assigned_clusters, points_in_cluster


	def K_Means_util(self, centroids):
		previous_centroids = centroids

		# calculate distance of each points from centroids
		distance_from_centroid = [] # len = n, each element in this list is a list of k length
		for i in self.data:
			dis = []
			for centroid in centroids:
				dis.append(self.dist(i, centroid))
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


	def find_best_K(self):
		fig = plt.figure(figsize=(20,5))
		plt1 = fig.add_subplot(121) 
		plt2 = fig.add_subplot(122)
		sse = []
		time_taken = []
		k_values = []
		for K in range(1, 16):
			self.K = K
			centroid, assigned_clusters, points_in_cluster, time = self.K_Means_unlabelled()
			sse.append(self.calculate_sse(points_in_cluster))
			time_taken.append(time)
			k_values.append(self.K)
		#print(k_values, time_taken, sse)      
		plt1.plot(k_values, time_taken)
		plt1.set_xlabel('K(no. of clusters)') 
		plt1.set_ylabel('time taken in seconds') 
		plt1.set_title('Time taken vs number of clusters') 
		plt2.plot(k_values, sse)
		plt2.set_xlabel('K(no. of clusters)') 
		plt2.set_ylabel('sse') 
		plt2.set_title('sse vs number of clusters')
		plt.show() 


	def run_on_K(self, K):
		self.K = K
		centroid, assigned_clusters, points_in_cluster, time = self.K_Means_unlabelled()
		return self.classification_accuracy(assigned_clusters)