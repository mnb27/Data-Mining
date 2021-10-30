import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA




'''

Formulae:

# theta = [(mu_1, cov_1, P_C_1), (mu_2, cov_2, P_C_2), ..............., (mu_K, cov_K, P_C_K)]

# P(x_i | theta_j) = Multivariate_Normal(x_i | mu_j, cov_j)

# w_ij = P(C|x_i) = { P(x_i | theta_j) * P_C_j } / { summation(P(x_i | theta_j) * prob_C-j) } 

# mean_j = { summation(w_ij * x_i) } / summation(w_ij)

# sigma_j = { summation w_ij * (x_i - mu_j) * (x_i - mu_j).T } / summation(w_ij)

# P_C_j = { summation(w_ij) } / n

'''




class EM_class:
    def __init__(self, data, K, epsilon, colors):
        self.data = data
        self.K = K
        self.epsilon = epsilon
        self.colors = colors
    

    def random_initialiser(self):
        # initialise means
        mu = np.random.choice(self.data.flatten(), (self.K, self.data.shape[1])) # K * D

        # initialise covariance matrix (positive semidefinite)
        sigma = [] # each element in this list is D*D matrix, len(sigma) = K
        for itr in range(self.K):
            sigma.append(np.identity(self.data.shape[1]))
        sigma = np.array(sigma)

        # cluster wights (K*1)
        prob_c = np.ones((self.K)) / self.K # equal weightage
        
        return mu, sigma, prob_c

    
    def prepare_grid(self):
        min_x = np.min(self.data[...,0])-1
        max_x = np.max(self.data[...,0])+1
        min_y = np.min(self.data[...,1])-1
        max_y = np.max(self.data[...,1])+1
        x = []
        y = []
        steps = 200
        for i in range(steps):
            x.append(min_x + i*(max_x-min_x)/steps)
        for i in range(steps):
            y.append(min_y + i*(max_y-min_y)/steps)
        X, Y = np.meshgrid(np.array(x), np.array(y))
        return X,Y
    
    
    def visualization_grid(self):
        X,Y = self.prepare_grid()
        pos = np.array([X.flatten(), Y.flatten()]).T
        return pos


    def plot_clusters(self, iteration_no, mu, sigma, prob_c):
        P_x_given_theta = [] # likelihood
        pos = self.visualization_grid()
        for j in range(self.K):
            P_x_given_theta.append(multivariate_normal.pdf(x=pos, mean=mu[j], cov=sigma[j]))
        P_x_given_theta = np.array(P_x_given_theta)
        pred = np.argmax(P_x_given_theta, axis=0)
        fig = plt.figure(figsize=(16,5))
        plt1 = fig.add_subplot(121)
        plt2 = fig.add_subplot(122, projection='3d')
        plt1.set_title("iteration no."+str(iteration_no)+" clusters")
        plt2.set_title("iteration no."+str(iteration_no)+" probability density")
        for i in range(self.K):
            pred_id = np.where(pred == i)
            plt1.scatter(pos[pred_id[0],0], pos[pred_id[0],1], color=self.colors[i], marker='o')
        plt1.scatter(self.data[...,0], self.data[...,1], facecolors='yellow', edgecolors='none')
        for i in range(self.K):
            plt1.scatter(mu[j][0], mu[j][1], color=self.colors[j], marker='D')
        X,Y = self.prepare_grid()
        pdf = (np.dot((prob_c.reshape(len(prob_c),1)).T, P_x_given_theta))
        pdf = pdf.reshape(X.shape)
        #print(X.shape, Y.shape, pdf.shape)
        plt2.plot_surface(X, Y, pdf, cmap='YlGnBu')
        plt2.scatter(self.data[:,0], self.data[:,1], np.zeros((self.data[:,0]).shape), color='red')
        plt1.set_xlabel('u0')
        plt1.set_ylabel('u1')
        plt2.set_xlabel('u0')
        plt2.set_ylabel('u1')
        plt2.set_zlabel('PDF')
        plt.show()


    def EM_algo(self, print_plot_for_iterations, max_iterations=100):
        mu, sigma, prob_c = self.random_initialiser()
        for iteration in range(max_iterations+1):
            # plot clusters
            if iteration in print_plot_for_iterations:
                self.plot_clusters(iteration, mu, sigma, prob_c)

            #####################################################################################################################
            #####################################################################################################################
                                                              # Expectation #
            #####################################################################################################################
            #####################################################################################################################
            P_x_given_theta = [] # likelihood
            for j in range(self.K):
                P_x_given_theta.append(multivariate_normal.pdf(x=self.data, mean=mu[j], cov=sigma[j]))
            P_x_given_theta = np.array(P_x_given_theta) # K * n
            # * w_ij is actually in expectation step but for better code writability it is written in maximization step

            #####################################################################################################################
            #####################################################################################################################
                                                              # Maximization #
            #####################################################################################################################
            #####################################################################################################################
            w_ij = []
            prev_mu = mu
            temp_mu = np.random.choice(self.data.flatten(), (self.K, self.data.shape[1]))
            for j in range(self.K):
                sum_ = (np.sum([P_x_given_theta[i]*prob_c[i] for i in range(self.K)], axis = 0))

                #########################################################################################
                # w_ij = P(C|x_i) = { P(x_i | theta_j) * prob_C-j } / { summation(P(x_i | theta_j) * prob_C_j) } 
                #########################################################################################
                w_ij.append((P_x_given_theta[j]*prob_c[j]) / sum_ + 1e-5) # 1e-5 to avoid deno menator to be zero

                sum_w_ij = np.sum(w_ij[j] + 1e-5) # 1e-5 to avoid deno menator to be zero

                #########################################################################################
                # mean_j = { summation(w_ij * x_i) } / summation(w_ij)
                #########################################################################################
                temp_mu[j] = np.sum( w_ij[j].reshape(len(self.data), 1) * (self.data), axis = 0 ) # (np.sum(n*D, axis=0)) == D*1
                temp_mu[j] = temp_mu[j]/sum_w_ij

                #########################################################################################
                # sigma_j = { summation w_ij * (x_i - mu_j) * (x_i - mu_j).T } / summation(w_ij)
                #########################################################################################
                temp = np.linalg.norm(np.add(self.data,-1*temp_mu[j]))
                sigma[j] = np.dot((w_ij[j].reshape(len(self.data), 1)*(self.data - temp_mu[j])).T, (self.data - temp_mu[j]))
                sigma[j] = sigma[j]/sum_w_ij

                #########################################################################################
                # prob_C_j = { summation(w_ij) } / n
                #########################################################################################
                prob_c[j] = np.mean(w_ij[j])
            norm_mu = []
            for j in range(self.K):
                norm_mu.append(np.linalg.norm(temp_mu[j]-prev_mu[j]))
            sum_ = 0
            for i in range(len(norm_mu)):
                sum_ += norm_mu[i]
            mu = temp_mu
            if sum_ < self.epsilon:
                self.plot_clusters(iteration, mu, sigma, prob_c)
                print("itrations taken : ", iteration+1)
                break
        return mu, sigma, prob_c

def main():
    df = pd.read_csv('spiral_old.csv')
    data = df.to_numpy()

    # apply PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data[:,:-1])
    label = data[:,-1]

    #plot data
    # fig = plt.figure(figsize=(8,6))
    # plt.scatter(pca_data[:,0][label=='Iris-setosa'],pca_data[:,1][label=='Iris-setosa'],color='b',marker='o',cmap='YlGnBu')
    # plt.scatter(pca_data[:,0][label=='Iris-versicolor'],pca_data[:,1][label=='Iris-versicolor'],color='g',marker='o',cmap='YlGnBu')
    # plt.scatter(pca_data[:,0][label=='Iris-virginica'],pca_data[:,1][label=='Iris-virginica'],color='r',marker='o',cmap='YlGnBu')
    # plt.xlabel('u0')
    # plt.ylabel('u1')
    # plt.title('pca iris dataset')
    # plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # plt.show()

    obj = EM_class(data=pca_data, K=2, epsilon=0.000001, colors=['orange', 'tab:green', 'tab:cyan'])
    print_plot_for_iterations = [0,10,20,30,40,50,60,70]
    mu, sigma, prob_c = obj.EM_algo(print_plot_for_iterations)

if __name__ == "__main__":
    main()