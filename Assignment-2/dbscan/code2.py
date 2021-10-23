import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def dbscan(data, eps, minPts):
    corePoint = []  # Core points array
    reachable = []  # Reachble points
    cluster = [0] * len(data)  # Clusters number array
    tempVisited = []  # Template visited array
    visited = []  # Visited points
    cl = 1  # Cluster number
    point = -1
    for i in range(len(data)):
        nclus = 0
        if not i in visited:

            tempVisited.append(i)  # Visited spots by a point
            sign = 0  # Check search for new cluster
            visited.append(i)
            while len(tempVisited) != 0:
                n = 0  # neighbor point number by a point
                if sign == 0:
                    point = i
                else:
                    point = tempVisited.pop()
                    visited.append(point)
                    if point == i:
                        break
                noisy = []
                c = 0
                for j in range(len(data)):

                    if point != j:

                        if distance(data[point], data[j]) <= eps:
                            if not (j in reachable or j in visited):
                                tempVisited.append(j)
                                c += 1

                            if j in visited and cluster[j] == 0:
                                noisy.append(j)

                            n += 1

                if n >= minPts and len(tempVisited) + len(noisy) >= minPts:
                    sign = 1
                    nclus = 1
                    corePoint.append(point)
                    cluster[point] = cl
                    assingCluster(noisy, cluster, cl)
                    assingCluster(tempVisited, cluster, cl)
                    assignReachable(tempVisited, reachable)
                elif sign != 1:
                    cluster[point] = 0  # Noisy point
                    tempVisited = []
                else:
                    for j in range(c):
                        tempVisited.pop()
        if nclus == 1:  # If new cluster found cl num uptade
            cl += 1

    return cluster, reachable, corePoint


def plotRes(data, cluster):
    clusterNum = max(cluster)
    scatterColors = ['pink', 'blue', 'green', 'brown', 'red', 'purple', 'orange', 'yellow', 'lime', 'peru', 'olive',
                     'gold', 'turquoise', 'tomato', 'lightcoral', 'khaki', 'seagreen', 'violet']
    for i in range(clusterNum + 1):
        if (i == 0):
            color = 'black'  # Paint noisy points to black
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(len(data)):
            if cluster[j] == i:
                x1.append(data[j][0])
                y1.append(data[j][1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')


def distance(point1, point2):  # Calculate distance between two point
    return math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))


def assingCluster(temp, clusterArray, cluster):  # Assign cluster to points
    for i in range(len(temp)):
        clusterArray[temp[i]] = cluster


def assignReachable(temp, reachable):  # Assign points as a reachable
    for i in range(len(temp)):
        if not temp[i] in reachable:
            reachable.append(temp[i])


# data = [[4, 29], [7, 28], [6, 27], [3, 27], [4, 26], [2, 25], [5, 25], [7, 25], [3, 23], [6, 23], [2, 22], [15, 28],
#         [19, 12], [17, 11], [20, 11], [18, 10], [16, 9], [19, 9], [20, 9], [18, 8], [20, 8], [22, 7], [18, 6], [20, 6],
#         [22, 22], [4, 9], [4, 31]]

import pandas as pd
df = pd.read_csv('spiral_new.csv', delimiter=',')
# User list comprehension to create a list of lists from Dataframe rows
list_of_rows = [list(row[0:2]) for row in df.values]
data = list_of_rows

reachable = []
cluster = []
corePoint = []
cluster, reachable, corePoint = dbscan(data, .4, 10)
print(max(cluster))
plotRes(data, cluster)
plt.show()