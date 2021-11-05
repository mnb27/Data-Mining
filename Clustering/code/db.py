import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def DBScanAlgo(data, sz, epsilon, minpts):
	data = data.to_numpy()
	cluster = 0
	clusterLabels = np.zeros(sz)
	visMark = np.zeros(sz) # visited array
	noisePointLabel = -1
	for i in range(sz):
		if visMark[i]!=1:
			visMark[i]=1
			neighborPts, length = getNeighbours(data,sz, i, epsilon)
			if length<minpts:
				clusterLabels[i]=noisePointLabel
			else:
				#expandCluster
				cluster += 1
				clusterLabels[i]=cluster
				j=0
				while j<length:
					index = neighborPts[j]
					if visMark[index]==0:
						visMark[index]=1
						new_neighbor, length_ = getNeighbours(data, sz ,index, epsilon)
						if length_>=minpts:
							neighborPts = neighborPts+new_neighbor
							length = len(neighborPts)
					if clusterLabels[index]!=1:
						clusterLabels[index]=cluster
					j = j + 1
	return clusterLabels

def getNeighbours(data, sz, center, eps):
	neighborPts = []
	centerPt = data[center]
	distances = list() 
	for i in range(sz):
		dist = np.linalg.norm(centerPt-data[i])
		distances.append(dist)

	for i in range(sz):
		if(distances[i]<eps):
			neighborPts.append(i)
	nbSize = len(neighborPts)
	return neighborPts, nbSize

def parseData(file_path, dim):

	data = pd.read_csv(file_path,sep=',', header=None, engine='python')
	data = data.apply(pd.to_numeric, errors='coerce')
	colNameIdx = 1
	if file_path=='adult.csv':
		features =  data.iloc[colNameIdx:,0:dim-1]
		labels = data.iloc[colNameIdx:,dim-1]
	
	else: # spiral
		features =  data.iloc[colNameIdx:,0:dim-1]
		labels = data.iloc[colNameIdx:,dim-1]
	return features, labels

def getKColors(k):
	colours = []
	from random import randint
	for i in range(k+1):
		colours.append('#%06X' % randint(0, 0xFFFFFF))
	return colours

def plotData(data,labels,title):
	data = data.to_numpy()
	clusters = list(set(labels))
	num_cluster = len(clusters)
	colours = getKColors(num_cluster)

	plt.figure()
	labelIds = list(set(labels))
	npLabel = np.array(labels)
	axes = list()
	for label in labelIds:
		axes.append(np.where(npLabel==label))

	idx = 0
	for label in labelIds:
		leged = "C-" + str(int(label))
		clr = colours[int(label)]
		if label==-1.0:
			leged = 'Noise'
			clr = 'b'
		plt.scatter(data[axes[idx],0],data[axes[idx],1],color=clr,label=leged)
		idx+=1
	plt.legend()
	plt.title(title)
	plt.xlabel('Feature X')
	plt.ylabel('Feature Y')
	plt.show()
	return None

def main(): 
	datasets = ['spiral_old.csv', 'spiral.csv','adult.csv']
	dim = 3  # 6 for adult dataset
	dataset, actualLabels = parseData(datasets[0], dim)
	plotData(dataset, actualLabels, 'Actual Data')
	npData = dataset.to_numpy()
	N = len(npData)
	# DBSCAN
	counter = 1
	epsilons = [0.4]
	minPoints = [10]
	for eps in epsilons:
		 for minPts in minPoints:
			 print("Run ",counter," : ","Epsilon - ",eps," Min Points - ",minPts)
			 counter += 1
			 DBSCAN_res = DBScanAlgo(dataset, N, eps, minPts)
			 unique_labels = set()
			 for label in DBSCAN_res:
				 unique_labels.add(label)
			 plotData(dataset, DBSCAN_res, 'DBSCAN Clustering')
			 print("Clusters: ",unique_labels)
			#  print("Actual: ",actualLabels[:5])
			#  print("Predicted: ",DBSCAN_res[:5])

if __name__ == "__main__":
	main()