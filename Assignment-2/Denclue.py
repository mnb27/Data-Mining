import math
import numpy as np
import matplotlib.pyplot as plt

class DENCLUE:
	def __init__(self, data):
		self.data = data
		self.h = 0.6
		self.si = 10

	def Gaussian_kernel(self, x):
		# assume covariance matrix is identity matrix
		sum_ = 0
		for xi in self.data:
			sum_ += math.exp( -0.5 * ( ( (np.linalg.norm(x-xi)) / self.h ) ** 2 ) )
		return sum_ / (self.data.shape[0] * (math.pow(self.h*math.pow(2*math.pi, 0.5), self.data.shape[1])))

	def prepare_grid(self, step=100):
		min_x = np.min(self.data[...,0])-1
		max_x = np.max(self.data[...,0])+1
		min_y = np.min(self.data[...,1])-1
		max_y = np.max(self.data[...,1])+1
		x = []
		y = []
		steps = step
		for i in range(steps):
			x.append(min_x + i*(max_x-min_x)/steps)
		for i in range(steps):
			y.append(min_y + i*(max_y-min_y)/steps)
		X, Y = np.meshgrid(np.array(x), np.array(y))
		return X,Y,x,y

	def plot_pdf(self, step = 50):
		X, Y, x, y = self.prepare_grid(step)
		pdf = []
		for j in y:
			for i in x:
				pdf.append(self.Gaussian_kernel([i, j]))
		pdf = np.array(pdf)
		pdf = pdf.reshape(X.shape)
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(111, projection='3d')
		ax.set_title("probability density for h = " + str(self.h))
		ax.plot_surface(X, Y, pdf, cmap='YlGnBu')
		ax.scatter(self.data[:,0], self.data[:,1], np.zeros((self.data[:,0]).shape), color='red')
		ax.set_xlabel('u0')
		ax.set_ylabel('u1')
		ax.set_zlabel('PDF')
		plt.show()

	def plot_pdf_in_h_range(self, h_list, step=50):
		for h in h_list:
			self.h = h
			self.plot_pdf(step)


	def find_attractor(self, x):
		sum_ = round((self.Gaussian_kernel(x) * (self.data.shape[0]) * math.pow(self.h, self.data.shape[1])),2)
		num = np.zeros((1, self.data.shape[1]))
		for i in range(self.data.shape[0]):
			temp = (math.exp(-0.5*(((np.linalg.norm(x-self.data[i,:]))/self.h)**2))) / (math.pow(math.pow(2*math.pi,0.5),self.data.shape[1]))
			#print(temp, self.data[i,:])
			for d in range(self.data.shape[1]):
				num[0,d] = num[0,d] + (temp * (self.data[i,d]))
			for d in range(self.data.shape[1]):
				num[0,d] = round(num[0,d]/sum_, 2)               
		return num[0]

	def algo(self, h, si):
		self.h = h
		self.si = si
		Attractors = []
		Region = {}
		# find density attractors
		for x in self.data:
			x_star = self.find_attractor(x)
			#print(x, x_star)
			if self.Gaussian_kernel(x_star) > self.si:
				#print(x_star)
				Attractors.append(x_star)
				if str(x_star) in Region.keys():
					Region[str(x_star)].append(x)
				else:
					Region[str(x_star)] = [x]
		return self.merged_clusters(Attractors, Region)

	def merged_clusters(self, Attractors, Region):
		new_Attractors = []
		new_A = {}
		for i in Attractors:
			for j in Attractors:
				if not (i == j).all():
					check = 0
					points_in_ij = []
					for x in self.data:
						flag = 0
						for d in range(self.data.shape[1]):
							if ((i[d] - x[d]) * (j[d] - x[d])) > 0:
								flag = 1
						if (flag == 1) or (self.Gaussian_kernel(x) < self.si):
							check = 1
					if check == 1:
						if str(i) not in new_A.keys():
							new_Attractors.append(i)
							new_A[str(i)] = 1
						if str(j) not in new_A.keys():
							new_Attractors.append(j)
							new_A[str(j)] = 1
					else:
						temp = i
						for d in range(self.data.shape[1]):
							temp[d] = round((i[d] + j[d])/2 ,2)
						if str(temp) not in new_A.keys():
							new_Attractors.append(temp)
							new_A[str(temp)] = 1
		#print(len(new_A))
		if len(new_Attractors) == len(Attractors):
			return new_Attractors
		else:
			return self.merged_clusters(new_Attractors, Region)