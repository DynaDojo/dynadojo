import numpy as np

from ...abstractions import AbstractSystem
from collections import Counter


class EpidemicSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 group_status,
                 seed=None):

        super().__init__(latent_dim, embed_dim, seed)

        if not group_status:
            assert embed_dim == latent_dim

        self._rng = np.random.default_rng(seed)

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range
        self.group_status = group_status

    def create_model(self, x0):
        return
    

    def create_randomized_dict(self, counts):
        # Create a list to hold (key, value) pairs
        pairs = []
        
        # Populate the list with the correct number of (key, value) pairs for each value
        for value, count in enumerate(counts):
            for _ in range(count):
                pairs.append((len(pairs), value))

       
        # Convert the list of pairs into a dictionary
        randomized_dict = dict(pairs)

        return randomized_dict

    def count_vals(self, input_array):
        # Count the occurrences of each value in the array
        value_counts = Counter(input_array)
        
        # Convert the counts to a list
        counts_list = [value_counts[value] for value in sorted(value_counts)]

        return counts_list

   
    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        grouped_x0 = []
        for _ in range(n):
            if in_dist:
                x0.append(np.floor(self._rng.uniform(self.IND_range[0], self.IND_range[1], (self.latent_dim))).astype(int))

            else:
                x0.append(np.floor(self._rng.uniform(self.OOD_range[0], self.OOD_range[1], (self.latent_dim))).astype(int))
        
        if self.group_status:
            for i in range(n):
                grouped_x0.append(self.count_vals(x0[i]))
            grouped_x0 = np.array(grouped_x0)
            return grouped_x0 
  
        x0 = np.array(x0)
        return x0 

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (self.latent_dim))
        else:
            noise = np.zeros((self.latent_dim))

        def dynamics(x0):
            x0_dict = {}
            if self.group_status:
                x0_dict = self.create_randomized_dict(x0)

            else:
                for idx, x in enumerate(x0):
                    x0_dict[idx] = x
           
            self.create_model(x0_dict)

            iterations = self.model.iteration_bunch(timesteps)
            dX = []
            for iteration in iterations:
                if(self.group_status):
                    step = [val for _, val in iteration['node_count'].items()]
                    dX.append(step)
                else:
                    step = []
                    for idx in range(self.latent_dim):
                        if (idx in iteration["status"]):
                            step.append(iteration["status"][idx])
                        else:
                            step.append(dX[-1][idx])
                    dX.append([int(x) for x in (step + noise)])
            return dX

        if control is not None:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                data.append(sol)

        data = np.array(data)
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

