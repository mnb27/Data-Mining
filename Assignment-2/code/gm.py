import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kme

def ExpectationPart(X, ax, pi, mu, sigma, K):
	reqShape = (X.shape[0], K)
	clusterIds = np.array(range(K))
	W = np.zeros(reqShape)
	for k in clusterIds:
		W[:,k] = pi[k] * getMultivariateNormalPDF(X, mu[k,:], sigma[k])
	W_norm = np.sum(W, axis=ax)[:,np.newaxis]
	normalizedGamma = W / W_norm
	return normalizedGamma, W


def MaximizationPart(X, W, pi, mu, sigma, clusterIds):

	WSum = np.sum(W, axis=0)
	WMean = np.mean(W, axis=0)
	WMatrix = np.dot(W.T, X)

	pi = WMean
	mu = WMatrix / WSum[:,np.newaxis]

	for k in clusterIds:
		x = X - mu[k, :]
		xT = x.T
		diagGammaMatrix = np.matrix(np.diag(W[:,k]))
		sigma_k = xT * diagGammaMatrix * x
		sigma[k,:,:]=(sigma_k) / WSum[:,np.newaxis][k]

	return pi, mu, sigma

def getMultivariateNormalPDF(X_data, muV, sigmaV):
	from scipy.stats import multivariate_normal
	return multivariate_normal.pdf(X_data, muV, sigmaV)

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

def makeMeshGrid(min_x, max_x, min_y, max_y):
	x = list()
	y = list()
	steps = 250
	for i in range(steps):
		x.append(min_x + i*(max_x-min_x)/steps)
		y.append(min_y + i*(max_y-min_y)/steps)
	X, Y = np.meshgrid(np.array(x), np.array(y))
	pos = np.array([X.flatten(), Y.flatten()])
	return X,Y,pos.T

def plot_clusters(data, prob_c, mu, sigma, iteration_no):
	axis00, axis01 = np.min(data[...,0]), np.min(data[...,1])
	axis10, axis11 = np.max(data[...,0]), np.max(data[...,1])
	X,Y,pos = makeMeshGrid(axis00-1, axis10+1, axis01-1, axis11+1)

	P_x_given_theta = list()
	P_x_given_theta.append(getMultivariateNormalPDF(pos, mu[0], sigma[0]))
	P_x_given_theta.append(getMultivariateNormalPDF(pos, mu[1], sigma[1]))

	pdf = (np.dot((prob_c.reshape(len(prob_c),1)).T, np.array(P_x_given_theta)))
	pdf = pdf.reshape(X.shape)

	fig = plt.figure(figsize=(7,7))
	plot = fig.add_subplot(111, projection='3d')
	plot.set_title("Iteration no. "+str(iteration_no)+" - Probability Density")
	plot.set_xlabel('X')
	plot.set_ylabel('Y')
	plot.set_zlabel('Density Function PDF')
	plot.plot_surface(X, Y, pdf, cmap='Greys')
	sctX, sctY, vec = data[:,0], data[:,1], np.zeros((data[:,0]).shape) 
	plot.scatter(sctX, sctY, vec, color='blue')

	plt.show()

def EMAlgorithmGMM(dataset, K, KMeansOp, centroids, max_itr=300):
	covMatShape = (K, centroids.shape[1], centroids.shape[1])

	idsList = list()
	clusters = np.unique(KMeansOp)
	for cluster in clusters:
		idsList.append(np.where(KMeansOp == cluster))

	covI = np.zeros(covMatShape) # start params
	piI = np.zeros((K))	
	muI = centroids 


	k = 0
	for cluster in clusters:
		piI[k] = len(idsList[k][0])/len(KMeansOp)
		de_mean = dataset.iloc[idsList[k]] - muI[k,:]
		Nk = len(idsList[k][0])
		covI[k,:,:] = np.dot(piI[k] * de_mean.T, de_mean) / Nk
		k = k + 1
	
	data = dataset.to_numpy()
	mu = muI
	sigma = covI
	pi = piI
		
	itr = 0
	flag = True
	while(flag):
		W, orgGamma = ExpectationPart(data,1,pi,mu,sigma,K)
		clusterIds = np.array(range(K))
		new_pi, new_mu, new_sigma = MaximizationPart(data,W,pi,mu,sigma,clusterIds)
		# plot_clusters(data, pi, mu, sigma, itr)

		BreakCond1 = itr > max_itr
		BreakCond2 = ((new_pi-pi).all() and (new_mu-mu).all() and (new_sigma-sigma).all())

		if BreakCond1 or BreakCond2 :
			flag  = False
		else:
			pi = new_pi
			mu = new_mu
			sigma = new_sigma
			itr = itr + 1
		if(not flag): break
			
	labels = np.zeros((data.shape[0], K))
	mulNormpdfs = []
	for k in range(K):
		mulNormpdfs.append(getMultivariateNormalPDF(data, mu[k,:], sigma[k]))
	for k in range(K):
		labels[:,k] = pi[k] * mulNormpdfs[k]

	plot_clusters(data, pi, mu, sigma, max_itr)
	return clusters[labels.argmax(1)], mu

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


def main():
	datasets = ['spiral_old.csv', 'spiral.csv','adult.csv']
	dim = 3  # 6 for adult dataset
	dataset, actualLabels = parseData(datasets[0], dim)
	# print(dataset)
	# print(actualLabels)
	plotData(dataset, actualLabels, 'Actual Data')

	# GMM clustering
	K = 2 # number of clusters 
	KMeansOp, kmean_centroids = kme.KMeansHelper(dataset,K,[], True)
	GMM_res, GMM_centroids = EMAlgorithmGMM(dataset, K,KMeansOp,kmean_centroids.to_numpy())
	# print("GMM Result: ",GMM_res)
	# print("Centroids: ", GMM_centroids)
	print("GMM: k = ", K)
	plotData(dataset, GMM_res, 'EM-GMM Clustering')


if __name__ == "__main__":
	main()