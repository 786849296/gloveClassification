import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman') 

frameNum = 9
pcaDim = 8
dataDir = "dataGlove"

# created by mwj
def AFS_cluster(feature_vectors, n_clusters):
    # Set the parameters for AFS clustering
    max_iter = 100
    population_size = 20
    step_size = 0.1
    visual_range = 0.2

    # Initialize the population of artificial fish
    population = np.random.rand(population_size, n_clusters, feature_vectors.shape[1])
    population_fitness = np.zeros(population_size)

    # Iterate for a maximum number of iterations
    for iteration in range(max_iter):
        # Evaluate the fitness of each fish
        for i in range(population_size):
            centers = population[i]
            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1).fit(feature_vectors)
            population_fitness[i] = -kmeans.score(feature_vectors)

        # Sort the population based on fitness
        sorted_indices = np.argsort(population_fitness)

        # Update the position of each fish
        for i in range(population_size):
            fish = population[sorted_indices[i]]
            for j in range(n_clusters):
                # Calculate the visual center of the fish's neighbors
                neighbors = np.delete(population[sorted_indices], i, axis=0)
                visual_center = np.mean(neighbors[:, j], axis=0)

                # Calculate the step vector
                step_vector = step_size * (visual_center - fish[j])

                # Calculate the new position of the fish
                new_fish = fish.copy()
                new_fish[j] += step_vector

                # Check if the new position is within the visual range
                if np.linalg.norm(new_fish[j] - visual_center) < visual_range:
                    population[sorted_indices[i], j] = new_fish[j]

    # Get the extreme points of the final population
    kmeans = KMeans(n_clusters=n_clusters, init=population[sorted_indices[0]], n_init=1).fit(feature_vectors)
    extreme_points = kmeans.cluster_centers_
    extreme_point_indices = []
    for center in extreme_points:
        closest_index = np.argmin(np.linalg.norm(feature_vectors - center, axis=1))
        extreme_point_indices.append(closest_index)

    return extreme_points, extreme_point_indices

for classDir in os.listdir(dataDir):
    classDir = os.path.join(dataDir, classDir)
    for i in os.listdir(classDir):
        frames = []
        i = os.path.join(classDir, i)
        for frame in os.listdir(i):
            if frame.endswith('.csv'):
                frame = os.path.join(i, frame)
                frames.append(pd.read_csv(frame, header=None, index_col=False, usecols=range(16), dtype=np.uint16))
        frames = np.stack(frames, 0)
        framesPCA = frames.reshape(frames.shape[0], -1)
        # pca = PCA(pcaDim)
        # framesPCA = pca.fit_transform(frames)
        x_min = np.min(framesPCA)
        x_max = np.max(framesPCA)
        framesPCA = (framesPCA - x_min) / (x_max - x_min)
        kmeans = KMeans(n_clusters=frameNum, init='k-means++', n_init='auto').fit(framesPCA)
        # extreme_points, indices = AFS_cluster(frames, frameNum)
        # afs_kmeans = KMeans(n_clusters=frameNum, init=extreme_points, n_init=1)
        # labels = afs_kmeans.fit_predict(frames)
        with open(os.path.join(i, "keyFrame.txt"), "w") as f:
            for indice in kmeans.labels_:
                f.write(str(indice) + ' ')
            f.write('\n')
        # if os.path.basename(i) == "10":
        #     tsne = TSNE(perplexity=6).fit(framesPCA)
            
        #     markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'H', '>']
        #     cmap = plt.cm.get_cmap("Set1")
        #     # 创建一个散点图，对于 kmeans.labels_ 中的每个唯一值
        #     for i, marker in zip(np.unique(kmeans.labels_), markers):
        #         plt.scatter(tsne.embedding_[kmeans.labels_ == i, 0], tsne.embedding_[kmeans.labels_ == i, 1], 
        #                     marker=marker, color=cmap(i), label=f'Cluster {i}')
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['bottom'].set_visible(False)
            # plt.gca().spines['left'].set_visible(False)
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()

            # scatter = plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=kmeans.labels_, cmap="Set1")
            # plt.xlabel("t-SNE dimension 1")
            # plt.ylabel("t-SNE dimension 2")
            # plt.show()
            
            # for indice in labels:
            #     f.write(str(indice) + ' ')
            # f.write('\n')
            # for indice in indices:
            #     f.write(str(indice) + ' ')
            # f.write('\n')
        