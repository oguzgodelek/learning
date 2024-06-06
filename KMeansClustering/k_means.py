import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=300, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.centroids = np.array([])
        self.assignments = []
        self.error_history = []
        self._data = np.array([])

    def fit(self, data):
        if self.n_clusters > data.shape[0]:
            raise RuntimeError('Number of cluster is bigger than the number of points')
        self.dimension = data.shape[1]
        boundaries = data.agg(['min', 'max'])
        self._data = np.array(data.values)
        self._random_initialize_centroids(self.dimension, boundaries)
        self.assignments = [-1 for _ in range(data.shape[0])]
        for idx in range(self.max_iter):
            self.error_history.append(self._assign_points())
            if self.verbose:
                print(f'Iteration {idx+1}: {self.error_history[-1]}')
            if idx > 2 and np.abs(self.error_history[-1] - self.error_history[-2]) < 1e-5:
                break
            self._update_centroids()

    def _random_initialize_centroids(self, dimension, boundaries):
        self.centroids = np.random.random((self.n_clusters, dimension))
        self.centroids = np.array(
            [
                [
                    boundaries[idx]['min']+row[idx]*(boundaries[idx]['max']-boundaries[idx]['min'])
                    for idx in range(dimension)
                ]
                for row in self.centroids
            ])

    def _assign_points(self):
        centroid_distances = np.linalg.norm(self._data.reshape(self._data.shape[0], 1, self._data.shape[1]) -
                                            self.centroids.reshape(1, self.centroids.shape[0], self.centroids.shape[1]),
                                            axis=2)

        self.assignments = np.argmin(centroid_distances, axis=1)
        return np.mean(np.min(centroid_distances, axis=1))

    def _update_centroids(self):
        self.centroids = np.array(
            [
                np.mean(self._data[self.assignments == idx], axis=0)
                for idx in range(self.n_clusters)
            ]
        )

    def draw_2d_points(self):
        if self._data.shape[1] != 2:
            print("You cannot draw scatter plots for the data with high dimensions")
            return
        cluster_colours = np.random.normal(size=self.n_clusters)
        plt.scatter(list(map(lambda x: x[0], self._data)), list(map(lambda x: x[1], self._data)),
                    c=[cluster_colours[x] for x in self.assignments])
        plt.show()


if __name__ == "__main__":
    first_distribution = np.random.multivariate_normal((0, 0), ((2, 0), (0, 2)), 500)
    second_distribution = np.random.multivariate_normal((4, 4), ((2, 0), (0, 2)), 500)
    points = pd.DataFrame(np.concatenate((first_distribution, second_distribution)))
    cluster = KMeansClustering(n_clusters=2)
    cluster.fit(points)
    cluster.draw_2d_points()
