from dynadojo.wrappers import SystemChecker
from dynadojo.utils.complexity_measures import gp_dim, mse_mv, pca
import pandas as pd
import os

from dynadojo.systems.lorenz import LorenzSystem
all_systems = ["LorenzSystem"]
def system_selector (system, dimension, seed):
    if system == "LorenzSystem":
        return SystemChecker(LorenzSystem(latent_dim=dimension, embed_dim=dimension, seed=seed))

#prep dataframe specs
column_names = ["system", "D", "seed", "x0", "OOD", "timesteps", 
                "gp_dim", "mse_mv", "pca", "lyapunov_spectrum", "kaplan_yorke_dimension", "pesin"] # noise?
                
seeds = [2]
dimensions = [3, 5, 7] #, 9]
timesteps_list = [100, 500, 1000]#, 2500, 5000, 10000]
max_timesteps = 1000#0
data = []

save_interval = 10
int_counter = 0

file_path = 'docs/dyna_complexity_data.JSON'
if os.path.isfile(file_path):
    df_old = pd.read_json(file_path, orient='records', lines=True) #loads pre-existing data
else:
    pd.DataFrame(columns=column_names).to_json(file_path, orient='records', lines=True)
    df_old = pd.read_json(file_path, orient='records', lines=True) #loads newly created data file

for system_name in all_systems:

    print()
    print("### WORKING ON:", system_name, "###")
    print()

    for dimension in dimensions:
        for seed in seeds:
            system = system_selector(system_name, dimension, seed)

            x0 = system.make_init_conds(1)
            x = system.make_data(x0, timesteps=max_timesteps)
            x0 = x0[0]
            x = x[0]
            
            y0 = system.make_init_conds(1, in_dist=False)
            y = system.make_data(y0, timesteps=max_timesteps)
            y0 = y0[0]
            y = y[0]

            for timesteps in timesteps_list:

                print()
                print("     ### CURRENTLY: dimension = ", dimension, ", seed = ", seed, ", timestep =", 
                    timesteps, "###")
                print()

                if df_old.empty == False: # guard against indexing into empty file
                    exists = ((df_old['system'] == system_name) & (df_old['D'] == dimension) & 
                            (df_old['seed'] == seed) & (df_old['timesteps'] == timesteps)).any()
                    if exists: #skips calculation if already exists in pre-existing data
                        continue

                X = x[:timesteps]
                Y = y[:timesteps]

                data.append([system_name, dimension, seed, x0, False, timesteps, gp_dim(X), 
                            mse_mv(X), pca(X), None, None, None])
                data.append([system_name, dimension, seed, y0, True, timesteps, gp_dim(Y), 
                            mse_mv(Y), pca(Y), None, None, None])
                
            if int_counter % save_interval == 0:
                temp_df = pd.DataFrame(data, columns=column_names)
                temp_df.to_json(file_path, mode='a', orient='records', lines=True)
                data = []
            int_counter += 1

if data:
    temp_df = pd.DataFrame(data, columns=column_names)
    temp_df.to_json(file_path, mode='a', orient='records', lines=True)

df_unsorted = pd.read_json(file_path, orient='records', lines=True)
df = df_unsorted.sort_values(by=['system', 'seed', 'D', 'OOD'])
df.to_json(file_path, orient='records', lines=True)