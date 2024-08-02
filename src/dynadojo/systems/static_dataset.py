import numpy as np
import pywt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from ..abstractions import AbstractSystem


class StaticDatasetSystem(AbstractSystem):
    """
    System for handling static datasets formatted in a numpy three dimensional array of N trajectories, T timesteps, and D dimensions.
    """

    def __init__(self, latent_dim=3, embed_dim=3, data: np.ndarray = None, seed=None, in_dist_ratio=0.8):
        """
        Initialize the StaticDatasetSystem.

        Args:
            latent_dim (int): Dimension of the latent space.
            embed_dim (int): Dimension of the embedding space.
            data (np.ndarray): 3-D numpy array with shape (N, T, D), where N is the number of trajectories,
                               T is the number of timesteps, and D is the dimension of the data.
            seed (int, optional): Random seed. Defaults to None.
            in_dist_ratio (float, optional): Ratio of in-distribution data. Defaults to 0.8.
        """
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            raise ValueError("No data was inputted or an empty array was provided. Please provide a valid 3D numpy array.")
        
        if data.ndim != 3:
            raise ValueError("Data should be a 3-D numpy array.")
        
        data_dim = data.shape[2]

        if embed_dim != data_dim:
            print(f"Embed dimension {embed_dim} is different from the data dimension {data_dim}. Adjusting embed_dim to {data_dim}.")
            embed_dim = data_dim

        if latent_dim != data_dim:
            print(f"Latent dimension {latent_dim} is different from the data dimension {data_dim}. Adjusting latent_dim to {data_dim}.")
            latent_dim = data_dim
        
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.data = data
        self.in_dist_ratio = in_dist_ratio

        if in_dist_ratio != 1:
            self.in_dist_data, self.ood_data = self._split_dataset(in_dist_ratio=self.in_dist_ratio)
        else:
            self.in_dist_data = self.data
            self.ood_data = None

    def _extract_wavelet_features(self, wavelet_name='db1', level=3) -> np.ndarray:
        """
        Extracts wavelet features from the dataset.

        This method decomposes each trajectory using wavelet transforms and flattens the coefficients into feature vectors.

        Args:
            wavelet_name (str): The name of the wavelet to use for decomposition (default is 'db1').
            level (int): The level of decomposition (default is 3).

        Returns:
            np.ndarray: A 2D array of shape (N, M) where N is the number of trajectories and M is the length of the flattened feature vectors.
        """
        num_trajectories, _, dimension = self.data.shape

        all_features = []

        for i in range(num_trajectories):
            trajectory = self.data[i]
            trajectory_features = []

            for j in range(dimension):
                coeffs = pywt.wavedec(trajectory[:, j], wavelet_name, level=level)

                flat_coeffs = np.concatenate([coeff.flatten() for coeff in coeffs])
                trajectory_features.append(flat_coeffs)

            all_features.append(np.concatenate(trajectory_features))

        max_feature_length = max(len(f) for f in all_features)

        features = np.zeros((num_trajectories, max_feature_length))

        for i, f in enumerate(all_features):
            features[i, :len(f)] = f

        return features


    def _split_dataset(self, in_dist_ratio: float)-> tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataset into in-distribution (IND) and out-of-distribution (OOD) points.

        This method uses KMeans clustering to separate the data into IND and OOD points based on the given ratio.

        Args:
            in_dist_ratio (float): Ratio of in-distribution data to be used (between 0.5 and 1).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains the in-distribution data.
                - The second array contains the out-of-distribution data.
        """
        if not (.5 <= in_dist_ratio < 1):
            raise ValueError("in_dist_ratio must be between 0.5 and 1 (inclusive).")

        np.random.seed(self._seed)

        # Split data into IND and OOD points using Kmeans clustering on trajectory features
        features = self._extract_wavelet_features()

        num_trajectories = features.shape[0]
        num_in_dist = round(in_dist_ratio * num_trajectories)
        num_ood = num_trajectories - num_in_dist

        if num_ood <= 0:
            raise ValueError(f"Not enough data to generate OOD points. Use dataset with more trajectories or adjust in_dist_ratio.")

        kmeans = KMeans(n_clusters=2, random_state=self._seed)
        labels = kmeans.fit_predict(features)

        cluster_0_size = np.sum(labels == 0)
        cluster_1_size = np.sum(labels == 1)
        in_dist_cluster = 0 if cluster_0_size > cluster_1_size else 1
        smaller_cluster = 1 - in_dist_cluster

        in_dist_indices = np.where(labels == in_dist_cluster)[0]
        smaller_cluster_indices = np.where(labels == smaller_cluster)[0]

        if len(in_dist_indices) > num_in_dist:
            smaller_cluster_center = kmeans.cluster_centers_[smaller_cluster]
            in_dist_features = features[in_dist_indices]
            distances = euclidean_distances(in_dist_features, smaller_cluster_center.reshape(1, -1))

            num_to_move = len(in_dist_indices) - num_in_dist

            closest_indices = np.argsort(distances.flatten())[:num_to_move]
            ood_indices_to_move = in_dist_indices[closest_indices]
            in_dist_indices = np.setdiff1d(in_dist_indices, ood_indices_to_move)
            ood_indices = np.concatenate([smaller_cluster_indices, ood_indices_to_move])
        elif len(in_dist_indices) < num_in_dist:
            cluster_center = kmeans.cluster_centers_[in_dist_cluster]
            smaller_cluster_features = features[smaller_cluster_indices]
            distances = euclidean_distances(smaller_cluster_features, cluster_center.reshape(1, -1))

            num_to_move = num_in_dist - len(in_dist_indices)

            closest_indices = np.argsort(distances.flatten())[:num_to_move]
            in_dist_indices = np.concatenate([in_dist_indices, smaller_cluster_indices[closest_indices]])
            ood_indices = np.setdiff1d(smaller_cluster_indices, smaller_cluster_indices[closest_indices])
        else:
            ood_indices = smaller_cluster_indices

        assert len(in_dist_indices) == num_in_dist
        assert len(ood_indices) == num_ood

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

    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False) -> np.ndarray:
        num_trajectories, num_timesteps, _ = self.data.shape
        if timesteps > num_timesteps:
            raise ValueError(f"Requested timesteps ({timesteps}) exceed the available timesteps ({num_timesteps}).")
    
        n = init_conds.shape[0]

        matched_indices = []
        for init_cond in init_conds:
            for i in range(num_trajectories):
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