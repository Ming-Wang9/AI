import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# https://docs.python.org/3/library/csv.html
def load_data(filepath):
    data = []
    with open(filepath, mode='r') as file:
        readed_file = csv.DictReader(file)
        for row in readed_file:
            data.append(dict(row))
    return data

def calc_features(row): 
    x1 = float(row.get('Population', 0))
    x2 = float(row.get('Net migration', 0))
    x3 = float(row.get('GDP ($ per capita)', 0))
    x4 = float(row.get('Literacy (%)', 0))
    x5 = float(row.get('Phones (per 1000)', 0))
    x6 = float(row.get('Infant mortality (per 1000 births)', 0))

    feature_vector = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    return feature_vector
    
# Compute the Euclidean distance between two vectors. get help from following links
#https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def hac(features):
    n = len(features)
    clusters = {i: [i] for i in range(n)}
    distance_matrix = np.zeros((n, n))

    # a symmetric distance matrix
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = euclidean_distance(features[i], features[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    
    Z = []
    
    for _ in range(n - 1):
        min_dist = np.inf
        closest_pair = None
        
        #closest pair of clusters
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < min_dist:
                    min_dist = distance_matrix[i, j]
                    closest_pair = (i, j)
        
        i, j = closest_pair
        new_cluster = n  

        clusters[new_cluster] = clusters[i] + clusters[j]
        Z.append([i, j, min_dist, len(clusters[new_cluster])])

        new_distance_matrix = np.inf * np.ones((n + 1, n + 1))

        for k in range(n):
            if k != i and k != j:
                new_distance_matrix[k, :n] = distance_matrix[k, :n]

        for k in range(n):
            if k != i and k != j:
                new_dist = min(distance_matrix[k, i], distance_matrix[k, j])
                new_distance_matrix[k, new_cluster] = new_dist
                new_distance_matrix[new_cluster, k] = new_dist
        
        new_distance_matrix[i, :] = np.inf
        new_distance_matrix[:, i] = np.inf
        new_distance_matrix[j, :] = np.inf
        new_distance_matrix[:, j] = np.inf
        
        distance_matrix = new_distance_matrix[:n + 1, :n + 1]
        n += 1  
    Z = np.array(Z, dtype=np.float64)
    return Z


def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    feature_matrix = np.array(features)  
    
    col_min = feature_matrix.min(axis=0)  
    col_max = feature_matrix.max(axis=0)  
    
    range_ = col_max - col_min
    range_[range_ == 0] = 1  
    normalized_matrix = (feature_matrix - col_min) / range_
    
    normalized_features = [normalized_matrix[i, :] for i in range(len(features))]
    
    return normalized_features


if __name__ == "__main__":
    data = load_data('countries.csv')
    features = [calc_features(row) for row in data]
    normalized_features = normalize_features(features)
    Z = hac(normalized_features)
    country_names = [row['Country'] for row in data]
    fig_hac(Z, country_names)

