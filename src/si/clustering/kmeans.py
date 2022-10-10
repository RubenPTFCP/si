class KMeans
    def __init__(self, k: int, max_iter: int = 1000):
        self.k=k
        self.distance=distance


    def _init_centroids(self, dataset: Dataset):
        seeds = np.random.permutation(dataset.shape()[0])[self.k]
        self.centroids = dataset.X[seeds]

    def distance

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis = 0)
        return closest_centroid_index
