#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kmeans
import math
from scipy.stats import multivariate_normal

def est_mult_gaus(X,mu,sigma):
    m = len(mu)
    sigma2 = np.diag(sigma)
    X = X-mu.T
    p = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma2)**(0.5))*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma2))*X,axis=1))

    return p

def e_step(X,pi,mu,sigma,n_clusters):
    N = X.shape[0] 
    gamma = np.zeros((N, n_clusters))
    const_c = np.zeros(n_clusters)    
    for c in range(n_clusters):
        gamma[:,c] = pi[c] * multivariate_normal.pdf(X, mu[c,:], sigma[c])
    # normalize across columns to make a valid probability
    gamma_norm = np.sum(gamma, axis=1)[:,np.newaxis]
    gamma /= gamma_norm
    
    return gamma


def m_step(X,gamma,pi,mu,sigma,n_clusters):
        N = X.shape[0]
        dim = X.shape[1]
        pi = np.mean(gamma, axis = 0)
        mu = np.dot(gamma.T, X) / np.sum(gamma, axis = 0)[:,np.newaxis]

        for c in range(n_clusters):
            x = X - mu[c, :]            
            gamma_diag = np.diag(gamma[:,c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)
            sigma_c = x.T * gamma_diag * x
            sigma[c,:,:]=(sigma_c) / np.sum(gamma, axis = 0)[:,np.newaxis][c]

        return pi, mu, sigma

def prepare_grid(data):
    min_x = np.min(data[...,0])-1
    max_x = np.max(data[...,0])+1
    min_y = np.min(data[...,1])-1
    max_y = np.max(data[...,1])+1
    x = []
    y = []
    steps = 200
    for i in range(steps):
        x.append(min_x + i*(max_x-min_x)/steps)
    for i in range(steps):
        y.append(min_y + i*(max_y-min_y)/steps)
    X, Y = np.meshgrid(np.array(x), np.array(y))
    return X,Y

def visualization_grid(data):
    X,Y = prepare_grid(data)
    pos = np.array([X.flatten(), Y.flatten()]).T
    return pos

def plot_clusters(iteration_no, mu, sigma, prob_c, data, colors=['orange', 'tab:green', 'tab:cyan']):
    P_x_given_theta = [] # likelihood
    pos = visualization_grid(data)
    for j in range(2):
        P_x_given_theta.append(multivariate_normal.pdf(x=pos, mean=mu[j], cov=sigma[j]))
    P_x_given_theta = np.array(P_x_given_theta)
    pred = np.argmax(P_x_given_theta, axis=0)
    fig = plt.figure(figsize=(16,5))
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122, projection='3d')
    plt1.set_title("iteration no."+str(iteration_no)+" clusters")
    plt2.set_title("iteration no."+str(iteration_no)+" probability density")
    for i in range(2):
        pred_id = np.where(pred == i)
        plt1.scatter(pos[pred_id[0],0], pos[pred_id[0],1], color=colors[i], marker='o')
    plt1.scatter(data[...,0], data[...,1], facecolors='yellow', edgecolors='none')
    for i in range(2):
        plt1.scatter(mu[j][0], mu[j][1], color=colors[j], marker='D')
    X,Y = prepare_grid(data)
    pdf = (np.dot((prob_c.reshape(len(prob_c),1)).T, P_x_given_theta))
    pdf = pdf.reshape(X.shape)
    #print(X.shape, Y.shape, pdf.shape)
    plt2.plot_surface(X, Y, pdf, cmap='YlGnBu')
    plt2.scatter(data[:,0], data[:,1], np.zeros((data[:,0]).shape), color='red')
    plt1.set_xlabel('u0')
    plt1.set_ylabel('u1')
    plt2.set_xlabel('u0')
    plt2.set_ylabel('u1')
    plt2.set_zlabel('PDF')
    plt.show()

def GMM_clustering(dataset,n_clusters,max_itr=1000):
    kmean_res, kmean_centroids = kmeans.k_means(dataset,n_clusters)
    data = dataset.to_numpy()
    centroids = kmean_centroids.to_numpy()
    datadim = centroids.shape[1]
    clusters = np.unique(kmean_res)
    
    #initialize parameters
    initial_means = centroids
    initial_cov = np.zeros((n_clusters,datadim,datadim))
    initial_pi = np.zeros((n_clusters))
    
    ct = 0
    for cluster in clusters:
        ids = np.where(kmean_res == cluster)
        initial_pi[ct] = len(ids[0])/len(kmean_res)
        de_mean = dataset.iloc[ids] - initial_means[ct,:]
        Nk = len(ids[0])
        initial_cov[ct,:,:] = np.dot(initial_pi[ct] * de_mean.T, de_mean) / Nk
        ct += 1
    
    pi = initial_pi
    mu = initial_means
    sigma = initial_cov
        
    itr = 0
    while(True):
        gamma = e_step(data,pi,mu,sigma,n_clusters)
        new_pi, new_mu, new_sigma = m_step(data,gamma,pi,mu,sigma,n_clusters)
        # plot_clusters(itr,mu,sigma,pi,data)
        if itr > max_itr or ((new_pi-pi).all() and (new_mu-mu).all() and (new_sigma-sigma).all()):
            break
        else:
            pi = new_pi
            mu = new_mu
            sigma = new_sigma
            itr += 1
            
    labels = np.zeros((data.shape[0], n_clusters))
    for c in range(n_clusters):
        labels[:,c] = pi[c] * multivariate_normal.pdf(data,mu[c,:],sigma[c])
    labels = labels.argmax(1)
    labels = clusters[labels]
    plot_clusters(max_itr,mu,sigma,pi,data)
    return labels, mu

def read(file_path):

    if file_path=='test.txt':
        data = pd.read_csv(file_path,sep=r'\t', header=None, engine='python')
        ds =  data.iloc[:,2:]
        gt = data.iloc[:,1]

    elif file_path=='adult.csv' or file_path=='test.csv':
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

# import pca as mypca
def visualize(dataset, labels, title):
    # pca_res = mypca.pca(dataset.to_numpy())
    visualization(dataset.to_numpy(), labels, title, 'PC')

def main():

    dataset1 =  'spiral_old.csv'
    dataset2 = 'spiral.csv'
    dataset3 = 'adult.csv'
    dataset4 = 'test.csv'
    dataset, ground_truth = read(dataset2)
    # print(dataset)
    # print(ground_truth)
    visualize(dataset, ground_truth, 'groundtruth')

    # GMM clustering
    k_clusters = 2
    GMM_res, GMM_centroids = GMM_clustering(dataset, k_clusters)
    # print("GMM Result: ",GMM_res)
    # print("Centroids: ", GMM_centroids)
    print('GMM: k = {}'.format(k_clusters))
    visualize(dataset, GMM_res, 'GMM')


if __name__ == "__main__":
    main()