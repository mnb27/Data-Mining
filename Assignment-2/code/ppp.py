import numpy as np
import random
import pandas as pd

class KMeansClass:
	def __init__(self, data, dim, K):
		self.K = K
		self.data = data[:, 0:dim]
		self.labels = data[:,dim- 1]
		
	def cluster_centroid(self, pointsInCluster):
		return np.mean(pointsInCluster, axis=0)

	def calcDistance(self, x, y):
		return np.linalg.norm(x-y)

	# Sum of squared error (SSE)
	def calcObjectiveCost(self, pointsInCluster, centroids):
		for i in range(self.K):
			ctd = self.cluster_centroid(pointsInCluster[i])
			centroids.append(ctd)

		SSE = 0 # objective cost		
		centroids = np.array(centroids)
		for cluster_no in pointsInCluster.keys():
			points = np.array(pointsInCluster[cluster_no])
			for point in points:
				# print("Debug --- ",point,"---",centroids[cluster_no])
				SSE = SSE + self.calcDistance(point[:-1], centroids[cluster_no][:-1])
		return SSE


	def closestPointToCentroid(self, distFromCentroid):
		pointsInCluster = {}
		for i in range(self.K):
			pointsInCluster[i] = [] 
		AssignedClusters = []
		for dist_list in distFromCentroid:
			AssignedClusters.append( dist_list.index(min(dist_list)) )
		for i in range(len(AssignedClusters)):
			pointsInCluster[AssignedClusters[i]].append(self.data[i])
		return AssignedClusters, pointsInCluster


	def HelperKMeans(self, centroids, sliceIdx):

		# calculate distance of each points from centroids
		distFromCentroid = []
		temp_dist = []
		for i in self.data[:,0:sliceIdx-1]:
			for centroid in centroids:
				# print(i,"---",centroid[0:sliceIdx-1])
				temp_dist.append(self.calcDistance(i, centroid[0:sliceIdx-1]))
			distFromCentroid.append(temp_dist)
			temp_dist = []

		prevCentroids = centroids
		AssignedClusters, pointsInCluster = self.closestPointToCentroid(distFromCentroid)

		# update cluster centroid
		newCentroids = np.zeros((self.K, self.data.shape[1]))
		for cluster_no, clusterr in pointsInCluster.items():
			newCentroids[cluster_no] = self.cluster_centroid(clusterr)

		toContinue = True
		if np.array_equal(prevCentroids, newCentroids):
			toContinue = False
		else: # repeat
			return self.HelperKMeans(newCentroids, sliceIdx=6)
		
		if not toContinue:
			return newCentroids, AssignedClusters, pointsInCluster

	def K_Means(self, OnInit):
		if OnInit: # Initialization of cluster centroids randomly
			centroids = np.zeros((self.K, self.data.shape[1])) # K * D
			temp = random.sample(range(0, self.data.shape[0]), self.K)
			for i in range(len(temp)):
				centroids[i] = self.data[temp[i]]
			OnInit = False

		FinalCentroids, FinalAssignedClusters, FinalpointsInCluster = self.HelperKMeans(centroids, sliceIdx=6)
		return FinalCentroids, FinalAssignedClusters, FinalpointsInCluster

	def startKMeans(self, K):
		self.K = K
		centroid, AssignedClusters, pointsInCluster = self.K_Means(True)
		return centroid, AssignedClusters, pointsInCluster, list(self.labels)

def main():
	# df = pd.read_csv('adult.csv')
	df = pd.read_csv('test.csv')
	data = df.to_numpy()
	# print("DATA: ", data)
	# print()

	k_list = [2, 5, 10, 20]
	KTaken = k_list[0]
	objectiveCosts = []
	balanceList = []
	maxDList = []
	for KTaken in k_list:
		obj = KMeansClass(data, data.shape[1], KTaken)
		res = obj.startKMeans(KTaken)
		
		# print("Centroid: ",res[0])
		# print()
		print("Assigned Clusters: ",res[1])
		print()
		# print("Points in cluster: ",res[2])
		# print()
		# print("Labels: ",res[3])
		objectiveCost = obj.calcObjectiveCost(res[2], [])
		print("SSE: ",objectiveCost)
		label = res[1]
		gender = res[3]

		#### GROUP FAIRNESS NOTION
		freqMale = {}
		freqFemale = {}
		for item in set(label):
			freqMale[item] = 0
			freqFemale[item] = 0

		for i in range(len(label)):
			item = label[i]
			if gender[i]==1.0:
				freqMale[item] += 1
			else :
				freqFemale[item] += 1

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
		for id in range(KTaken):
			fartestDistMalePoint = float('-inf')
			fartestDistFemalePoint = float('-inf')
			# print(id," --- ", pointsInCluster[id])
			for point in pointsInClusterwithLabel[id]:
				gender = point[5]
				distance = obj.calcDistance(clusterCentroids[id], point[:-1])
				if gender==1.0 and distance > fartestDistMalePoint:
					fartestDistMalePoint = distance
				elif gender==0.0 and distance > fartestDistFemalePoint:
					fartestDistFemalePoint = distance    
			if fartestDistFemalePoint!=0.0: p.append(fartestDistMalePoint/fartestDistFemalePoint)
			else: p.append(float('inf'))
		print("P array: ",p)
		maxD = min(p)
		print("MaxD: ",maxD)
		objectiveCosts.append(objectiveCost)
		balanceList.append(balance)
		maxDList.append(maxD)
		# print(clusterCentroids[:,0:5])
		# print(pointsInClusterwithLabel[0][1][5])
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8,7))
	plt.plot(k_list, balanceList, label = "Balance")
	plt.plot(k_list, maxDList, label = "maxD")
	plt.xlabel('k-clusters') 
	plt.ylabel('Fairness Value') 
	plt.title('Variation of fairness metrics - balance and maxD over different k values') 
	plt.legend()
	plt.show()

	plt.figure(figsize=(8,7))
	plt.plot(k_list, objectiveCosts, label = "Objective Cost")
	plt.xlabel('k-clusters') 
	plt.ylabel('Cost') 
	plt.title('Variation of objective cost over different k values') 
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()