import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from ..abstractions import AbstractSystem


class StaticDatasetSystem(AbstractSystem):
    """
    System for handling static datasets formatted in a numpy three dimensional array of (N trajectories, T timesteps, and D dimensions).
    """

    def __init__(self, latent_dim, embed_dim, data: np.ndarray, seed=None, in_dist_ratio=0.8):
        """
        Initialize the StaticDatasetSystem.

        Args:
            latent_dim (int): Dimension of the latent space.
            embed_dim (int): Dimension of the embedding space.
            data (np.ndarray): 3-D numpy array of shape (N, T, D) where N is the number of trajectories,
                               T is the number of timesteps, and D is the dimension of the data.
            seed (int, optional): Random seed. Defaults to None.
            in_dist_ratio (float, optional): Ratio of in-distribution data. Defaults to 0.8.
        """
        if data.ndim != 3:
            raise ValueError("Data should be a 3-D numpy array.")
        
        self.trajectories, self.timesteps, D = data.shape
        if embed_dim != D:
            raise ValueError(f"Embed dimension should be equal to the dimension of the data. Expected {D}, but got {embed_dim}.")
        
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.data = data
        self.in_dist_ratio = in_dist_ratio

        if in_dist_ratio != 1:
            self.in_dist_data, self.ood_data = self.split_dataset(in_dist_ratio=self.in_dist_ratio)
        else:
            self.in_dist_data = self.data
            self.ood_data = None

    def split_dataset(self, in_dist_ratio: float):
        if not (.5 < in_dist_ratio < 1):
            raise ValueError("in_dist_ratio must be between 0.5 and 1")

        np.random.seed(self._seed)

        flattened_data = self.data.reshape(self.trajectories, -1)

        num_in_dist = round(in_dist_ratio * self.trajectories)
        num_ood = self.trajectories - num_in_dist

        if num_ood <= 0:
            raise ValueError(f"Not enough data to generate OOD points. Use dataset with more trajectories or adjust in_dist_ratio.")

        kmeans = KMeans(n_clusters=2, random_state=self._seed)
        labels = kmeans.fit_predict(flattened_data)

        cluster_0_size = np.sum(labels == 0)
        cluster_1_size = np.sum(labels == 1)
        in_dist_cluster = 0 if cluster_0_size > cluster_1_size else 1
        smaller_cluster = 1 - in_dist_cluster

        in_dist_indices = np.where(labels == in_dist_cluster)[0]
        smaller_cluster_indices = np.where(labels == smaller_cluster)[0]

        in_dist_indices = np.where(labels == in_dist_cluster)[0]
        smaller_cluster_indices = np.where(labels == smaller_cluster)[0]

        if len(in_dist_indices) > num_in_dist:
            smaller_cluster_center = kmeans.cluster_centers_[smaller_cluster]
            in_dist_flattened = flattened_data[in_dist_indices]
            distances = euclidean_distances(in_dist_flattened, smaller_cluster_center.reshape(1, -1))

            num_to_move = len(in_dist_indices) - num_in_dist

            closest_indices = np.argsort(distances.flatten())[:num_to_move]
            ood_indices_to_move = in_dist_indices[closest_indices]
            in_dist_indices = np.setdiff1d(in_dist_indices, ood_indices_to_move)
            ood_indices = np.concatenate([smaller_cluster_indices, ood_indices_to_move])
        elif len(in_dist_indices) < num_in_dist:
            cluster_center = kmeans.cluster_centers_[in_dist_cluster]
            smaller_cluster_flattened = flattened_data[smaller_cluster_indices]
            distances = euclidean_distances(smaller_cluster_flattened, cluster_center.reshape(1, -1))

            num_to_move = num_in_dist - len(in_dist_indices)

            closest_indices = np.argsort(distances.flatten())[:num_to_move]
            in_dist_indices = np.concatenate([in_dist_indices, smaller_cluster_indices[closest_indices]])
            ood_indices = np.setdiff1d(smaller_cluster_indices, smaller_cluster_indices[closest_indices])
        else:
            ood_indices = smaller_cluster_indices

        assert len(in_dist_indices) == num_in_dist
        assert len(ood_indices)== num_ood

        np.random.shuffle(ood_indices)
        np.random.shuffle(in_dist_indices)

        in_dist_data = self.data[in_dist_indices]
        ood_data = self.data[ood_indices]

        return in_dist_data, ood_data

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        np.random.seed(self._seed)
        
        if in_dist:
            data_source = self.in_dist_data
        else:
            data_source = self.ood_data
        
        if data_source is None:
            raise ValueError(f"Cannot generate OOD points because in_dist_ratio is set to 1.")

        num_trajectories = data_source.shape[0]

        if n > num_trajectories:
            raise ValueError(f"Requested {n} initial conditions, but only {num_trajectories} trajectories available.")
        
        random_indices = np.random.choice(num_trajectories, size=n, replace=False)
        x0 = data_source[random_indices, 0, :]
        
        return x0

    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False):
        if timesteps > self.timesteps:
            raise ValueError(f"Requested timesteps ({timesteps}) exceed the available timesteps ({self.timesteps}).")
    
        n = init_conds.shape[0]
        
        matched_indices = []
        for init_cond in init_conds:
            for i in range(self.trajectories):
                if np.allclose(self.data[i, 0, :], init_cond):
                    matched_indices.append(i)
                    break
        
        assert len(matched_indices) == n, "Could not find matching trajectories for all initial conditions"
        
        x = np.array([self.data[idx, :timesteps, :] for idx in matched_indices])
        
        return x

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self._embed_dim