from dynadojo.wrappers import SystemChecker
from dynadojo.utils.complexity_measures import gp_dim, mse_mv, pca, find_lyapunov_exponents, find_max_lyapunov, kaplan_yorke_dimension, pesin

import pandas as pd
import os
import dysts
import json

from dynadojo.systems.gilpin_flows import GilpinFlowsSystem

# pull Gilpin's system info json
base_path = os.path.dirname(dysts.__file__)
json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')
with open(json_file_path, 'r') as file:
    systems_data = json.load(file)
all_systems = list(systems_data.keys())

# Systems that crashed the code when generating some seeds
problematic_systems = ["IkedaDelay", "MackeyGlass", "PiecewiseCircuit", "RabinovichFabrikant", 
                       "ScrollDelay", "SprottDelay", "SprottJerk", "SprottL", "SprottM", "SprottQ", "VossDelay", "Torus", "TurchinHanski"]

for problematic_system in problematic_systems:
    all_systems.remove(problematic_system)

# prep dataframe columns
column_names = ["system", "D", "seed", "x0", "OOD", "timesteps", 
                "gp_dim", "mse_mv", "pca", "lyapunov_spectrum", "kaplan_yorke_dimension", "pesin"] # noise?

# NOTE: specify sweep parameters here, as well as how often the code should save to JSON
seeds = [1, 2, 3, 4, 5]
dimensions = [3]
timesteps_list = [100, 500, 1000, 2500]
max_timesteps = 2500

data = []
save_interval = 2 # Save every "save_interval" number of seeds
int_counter = 0

# Checkpointing system to reduce redundancy if partial data already exists
file_path = 'docs/gilpin_complexity_data.JSON'
if os.path.isfile(file_path):
    df_old = pd.read_json(file_path, orient='records', lines=True) # loads pre-existing data
else:
    pd.DataFrame(columns=column_names).to_json(file_path, orient='records', lines=True)
    df_old = pd.read_json(file_path, orient='records', lines=True) # loads newly created data file

# Loop to sweep and generate code
for system_name in all_systems:
    print()
    print("### WORKING ON:", system_name, "###")
    print()
    for dimension in dimensions:
        for seed in seeds:
            system = SystemChecker(GilpinFlowsSystem(latent_dim=dimension, embed_dim=dimension, system_name=system_name, seed=seed))
            unwrapped_system = system._system # Reach under SystemChecker to enable "return_times" kwarg to GilpinFlowsSystem's "make_data" method
            model = unwrapped_system.system # Reach under GilpinFlowsSystem to access the model for Lyapunov

            # Generate in distribution trajectory data
            x0 = system.make_init_conds(1)
            xtpts, x = unwrapped_system.make_data(x0, timesteps=max_timesteps, return_times=True)
            x0 = x0[0]
            x = x[0]
            
            # Generate out of distribution trajectory data
            y0 = system.make_init_conds(1, in_dist=False)
            ytpts, y = unwrapped_system.make_data(y0, timesteps=max_timesteps, return_times=True)
            y0 = y0[0]
            y = y[0]

            # Calculate the measures for each "timesteps" sub-slice of the overall trajectory
            for timesteps in timesteps_list:
                print()
                print("     ### CURRENTLY: dimension = ", dimension, ", seed = ", seed, ", timestep =", 
                      timesteps, "###")
                print()

                if df_old.empty == False: # guard against indexing into empty file
                    exists = ((df_old['system'] == system_name) & (df_old['D'] == dimension) & 
                            (df_old['seed'] == seed) & (df_old['timesteps'] == timesteps)).any()
                    if exists: # skips calculation if already exists in pre-existing data
                        continue

                # Aforementioned subsets of trajectories
                X = x[:timesteps] 
                Y = y[:timesteps]
                
                xlyapunov_spectrum = find_lyapunov_exponents(X, xtpts, timesteps, model)
                ylyapunov_spectrum = find_lyapunov_exponents(Y, ytpts, timesteps, model)
                data.append([system_name, dimension, seed, x0, False, timesteps, gp_dim(X), 
                             mse_mv(X), pca(X), xlyapunov_spectrum, 
                             kaplan_yorke_dimension(xlyapunov_spectrum), pesin(xlyapunov_spectrum)])
                data.append([system_name, dimension, seed, y0, True, timesteps, gp_dim(Y), 
                             mse_mv(Y), pca(Y), ylyapunov_spectrum, 
                             kaplan_yorke_dimension(ylyapunov_spectrum), pesin(ylyapunov_spectrum)])
                
            # Code to append & save to existing JSON periodically
            if int_counter % save_interval == 0:
                temp_df = pd.DataFrame(data, columns=column_names)
                temp_df.to_json(file_path, mode='a', orient='records', lines=True)
                data = []
            
            int_counter += 1

# Check and append any remaining data
if data:
    temp_df = pd.DataFrame(data, columns=column_names)
    temp_df.to_json(file_path, mode='a', orient='records', lines=True)

# Re-sort the JSON
df_unsorted = pd.read_json(file_path, orient='records', lines=True)
df = df_unsorted.sort_values(by=['system', 'seed', 'D'])
df.to_json(file_path, orient='records', lines=True)