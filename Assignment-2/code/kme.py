import numpy as np

def assignPointsToClosestCentroids(df, centroids, N):
    dVectors = []
    maxVal = float('inf')
    for i in range(N):
        dVectors.append(df.iloc[i,:])
    
    closest = np.zeros(N)
    for i in range(N):
        min_dist = maxVal
        for centroid in range(centroids.shape[0]):
            dist = np.linalg.norm(dVectors[i] - centroids.iloc[centroid,:])
            if dist < min_dist:
                min_dist = dist
                closest[i] = centroid
    return closest

def KMeansHelper(dataset, k, clusters, onInit):

    if(onInit):
        # return randomly selected k points with its range
        x = np.random.choice(dataset.shape[0], k, False)
        centroids = dataset.iloc[x,:].reset_index(drop=True)

    onInit = False

    # assign points to clusters and update centroids : repeat until centroids no longer change
    flag = True
    while(flag):
        clusters = assignPointsToClosestCentroids(dataset, centroids, dataset.shape[0])
        new_c = dataset.groupby(clusters).mean()
        if new_c.equals(centroids):
            flag = False
        else:
            centroids = new_c
        if(not flag): break
    # format output
    clusters = clusters + 1
    clusters = clusters.astype(int)
    return clusters, centroids