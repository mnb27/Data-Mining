import numpy as np

def centeroidnp(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

data = [[0,1],[1,2]]
print(centeroidnp(np.array(data)))
