import tensorflow as tf
import numpy as np
import functions as func


n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed= seed)

centroids, samples = func.create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
from functions import create_samples, choose_random_centroids,  plot_clusters,  assign_to_nearest,  update_centroids

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)


model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)
    func.plot_clusters(sample_values,  updated_centroid_value,  n_samples_per_cluster)
