from sklearn.cluster import MiniBatchKMeans
import numpy as np

print("LOADING DATASET...")
dataset = np.loadtxt("./desc.txt")

print("CLUSTERING...")
kmeans = MiniBatchKMeans(n_clusters=1000, max_iter=1, batch_size=1024)
kmeans.fit(dataset)

print("SAVING...")
np.savetxt("./codebook.txt", kmeans.cluster_centers_)
