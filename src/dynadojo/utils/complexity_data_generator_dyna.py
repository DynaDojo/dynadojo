from dynadojo.wrappers import SystemChecker
from dynadojo.utils.complexity_measures import gp_dim, mse_mv, pca
import pandas as pd
# loads pre-existing data
df = pd.read_json('docs/dyna_complexity_data.JSON', orient='records', lines=True)
# specifies column names for panda dataframe
column_names = ["system", "D", "seed", "x0", "OOD", "timesteps", "gp_dim", "mse_mv", "pca", "max_lyapunov", "kaplan_yorke_dimension", "pesin"]
# loads dynadojo systems
from dynadojo.systems.lorenz import LorenzSystem
from dynadojo.systems.kuramoto import KuramotoSystem
from dynadojo.systems.lv import CompetitiveLVSystem, PreyPredatorSystem
from dynadojo.systems.santi import NBodySystem
all_systems = ["LorenzSystem", "KuramotoSystem", "CompetitiveLVSystem", "PreyPredatorSystem", "NBodySystem"]
def system_selector (system, dimension, seed):
    if system == "LorenzSystem":
        return SystemChecker(LorenzSystem(latent_dim=dimension, embed_dim=dimension, seed=seed))
    elif system == "KuramotoSystem":
        return SystemChecker(KuramotoSystem(latent_dim=dimension, embed_dim=dimension, seed=seed))
    elif system == "CompetitiveLVSystem":
        return SystemChecker(CompetitiveLVSystem(latent_dim=dimension, embed_dim=dimension, seed=seed))
    elif system == "PreyPredatorSystem":
        return SystemChecker(PreyPredatorSystem(latent_dim=dimension, embed_dim=dimension, seed=seed))
    elif system == "NBodySystem":
        return SystemChecker(NBodySystem(latent_dim=dimension, embed_dim=dimension, seed=seed))
# to store newly generated data
data = []
for system_name in all_systems:
    # USER'S CHOICE
    seeds = [1]
    if system_name == "NBodySystem":
        # USER'S CHOICE: Must be multiple of four
        dimensions = [4, 8, 12, 16]
    else:
        # USER'S CHOICE: Must be odd number above three
        dimensions = [3, 5, 7, 9]
    # USER'S CHOICE
    timesteps_list = [100, 500, 1000, 2500, 5000, 10000]
    max_timesteps = 10000
    print()
    print("### WORKING ON:", system_name, "###")
    print()
    for seed in seeds:
        for dimension in dimensions:
            system = system_selector(system_name, dimension, seed)
            # creates max length in-distribution trajectory
            x0 = system.make_init_conds(1)
            x = system.make_data(x0, timesteps=max_timesteps)
            x0 = x0[0]
            x = x[0]
            # creates max length out-of-distribution trajectory
            y0 = system.make_init_conds(1, in_dist=False)
            y = system.make_data(y0, timesteps=max_timesteps)
            y0 = y0[0]
            y = y[0]
            for timesteps in timesteps_list:
                print()
                print("     ### CURRENTLY: dimension =", dimension, ", seed =", seed, ", timesteps =", timesteps, "###")
                print()
                # skips calculation if already exists in pre-existing data
                exists = ((df['system'] == system_name) & (df['D'] == dimension) & (df['seed'] == seed) & (df['timesteps'] == timesteps)).any()
                if exists:
                    continue
                # slices out subset of max length trajectory
                X = x[:timesteps]
                Y = y[:timesteps]
                # calculates complexity measures, Lyapunov does not yet exist for dynadojo systems
                data.append([system_name, dimension, seed, x0, False, timesteps, gp_dim(X), mse_mv(X), pca(X), None, None, None])
                data.append([system_name, dimension, seed, x0, True, timesteps, gp_dim(Y), mse_mv(Y), pca(Y), None, None, None])
# merges and formats newly generated data and pre-existing data
df_new = pd.DataFrame(data, columns=column_names)
result = pd.concat([df, df_new], ignore_index=True)
result = result.sort_values(by=['system', 'seed', 'D'])
result.to_json('docs/dyna_complexity_data.JSON', orient='records', lines=True)