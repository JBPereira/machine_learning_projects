import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import random

"""
Function for sampling line item data. It picks random samples out of different clusters of points. In this way,
the sample should be more representative of the actual data
"""


def stratified_sampling(data, sample_size, min_n_clusters=5, max_n_clusters=15):

    """
    Create a representative sample of the data, capturing as much variation as possible
    :param data: Set of line items
    :param sample_size: Size of the desired sample
    :param min_n_clusters: minimum number of groups the points are to assigned to before sampling
    :param max_n_clusters: maximum " " "
    :return: Representative sample of the data
    """

    clustering_silhouette = 0
    norm_data = stats.zscore(data)
    outlier_list = []
    for i in range(len(norm_data)):
        for j in norm_data[i, :]:
            if j >= 2.8 or j <= -2.8:
                outlier_list.append(i)
                break
    norm_data = np.delete(norm_data, outlier_list, 0)

    for i in range(min_n_clusters, max_n_clusters):
        k_means = KMeans(i)
        k_means.fit(norm_data)
        current_silhouette = silhouette_score(norm_data, k_means.labels_)
        if current_silhouette > clustering_silhouette:
            clustering_silhouette = current_silhouette
            best_label = k_means.labels_

    selected_points = np.zeros((sample_size, np.shape(data)[1]))

    max_cluster = max(best_label)

    cluster_array = np.arange(max_cluster)

    random.shuffle(cluster_array)

    current_cluster_index = 0

    for i in range(sample_size):
        cluster_points = data[best_label == cluster_array[current_cluster_index], :]
        selected_points[i] = cluster_points[np.random.randint(0, len(cluster_points)), :]
        current_cluster_index += 1
        if current_cluster_index >= max_cluster:
            current_cluster_index = 0

    return selected_points
