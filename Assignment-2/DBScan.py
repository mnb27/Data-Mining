import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DBScan_class:
	def __init__(self, data):
		self.data = data[:,0:data.shape[1] - 1] # n*D
		self.labels = data[:,data.shape[1] - 1] # n*1
		self.epsilon = 1
		self.min_points = 2
		self.id = {}
		self.neighbour = {}
		self.core_points = []
		self.cluster_label = {}


	def density_connected(self, x_i, K):
		for y in self.neighbour[str(x_i)]:
			if self.id[str(y)] == 0:
				if not (y==x_i).all():
					self.id[str(y)] = K
					for i in self.core_points:
						if (y == i).all():
							#print(x_i, y, i)
							self.density_connected(y, K)


	def dist(self, x, y):
		sum_ = 0
		for d in range(len(x)):
			sum_ += (x[d] - y[d]) ** 2
		return math.sqrt(sum_)


	def classification_accuracy(self, cluster_labeling):
		#print(cluster_labeling)
		net_cluster_label = {}
		for cluster_no in cluster_labeling.keys():
			net_cluster_label[cluster_no] = {}
			for label in cluster_labeling[cluster_no]:
				if label in net_cluster_label[cluster_no].keys():
					net_cluster_label[cluster_no][label] += 1
				else:
					net_cluster_label[cluster_no][label] = 1
		#print(net_cluster_label)
		for cluster_no in net_cluster_label.keys():
			majority = 0
			for label in net_cluster_label[cluster_no].keys():
				if majority <= net_cluster_label[cluster_no][label]:
					self.cluster_label[cluster_no] = label
					majority = net_cluster_label[cluster_no][label]
		misclassified = 0
		for cluster_no in cluster_labeling.keys():
			for label in cluster_labeling[cluster_no]:
				if not (label == self.cluster_label[cluster_no]):
					misclassified += 1
		return round( ( (self.data.shape[0]-misclassified)*100 / (self.data.shape[0])), 2 )
    
    
	def DBScan_algo(self, epsilon, min_points):
		self.id = {}
		self.neighbour = {}
		self.core_points = []
		self.cluster_label = {}
		self.epsilon = epsilon
		self.min_points = min_points
		for x_i in self.data:
			#print(x_i)
			self.neighbour[str(x_i)] = []
			for y in self.data:
				if self.dist(x_i, y) <= self.epsilon:
					self.neighbour[str(x_i)].append(y)
			self.id[str(x_i)] = 0
			if len(self.neighbour[str(x_i)]) >= self.min_points:
				self.core_points.append(x_i)
		K = 0
		for x_i in self.core_points:
			if self.id[str(x_i)] == 0:
				K = K + 1
				self.id[str(x_i)] = K
				self.density_connected(x_i, K)
		clusters = {} # len of dictionary is equals to no. of clusters
		cluster_labeling = {}
		index = 0
		for x in self.data:
			if self.id[str(x)] > 0:
				if self.id[str(x)] not in clusters.keys():
					clusters[self.id[str(x)]] = [x]
					cluster_labeling[self.id[str(x)]] = [self.labels[index]]
				else:
					clusters[self.id[str(x)]].append(x)
					cluster_labeling[self.id[str(x)]].append(self.labels[index])
			index += 1
		return clusters, K, len(self.core_points), cluster_labeling


	def find_optimal_parameters(self, epsilons, min_points):
		fig = plt.figure(figsize=(8, 6))
		X = []
		Y = []
		Z1 = []
		Z2 = []
		Z3 = []
		for min_point in min_points:
			for epsilon in epsilons:
				st = datetime.now()
				clusters, K, core_points, _ = self.DBScan_algo(epsilon, min_point)
				ed = datetime.now()
				time = round((ed-st).total_seconds(), 2)
				X.append(epsilon)
				Y.append(min_point)
				Z1.append(K)
				Z2.append(core_points)
				Z3.append(time)
				#print(epsilon, min_point, K, core_points, time)
		plt1 = fig.add_subplot(111, projection='3d')
		plt1.plot_trisurf(np.array(X),np.array(Y),np.array(Z1))
		plt1.set_xlabel('epsilon') 
		plt1.set_ylabel('min_points') 
		plt1.set_zlabel('number of clusters')
		plt.show()
		fig = plt.figure(figsize=(8,6))
		plt2 = fig.add_subplot(111, projection='3d')
		plt2.plot_trisurf(np.array(X),np.array(Y),np.array(Z2))
		plt2.set_xlabel('epsilon') 
		plt2.set_ylabel('min_points') 
		plt2.set_zlabel('number of core points') 
		plt.show()
		fig = plt.figure(figsize=(8,6))
		plt3 = fig.add_subplot(111, projection='3d')
		plt3.plot_trisurf(np.array(X),np.array(Y),np.array(Z3))
		plt3.set_xlabel('epsilon') 
		plt3.set_ylabel('min_points') 
		plt3.set_zlabel('run time in seconds') 
		plt.show()


	def DBScan_diven_param(self, epsilon, min_point):
		clusters, K, core_points, cluster_labeling = self.DBScan_algo(epsilon, min_point)
		#print(K)
		return self.classification_accuracy(cluster_labeling)