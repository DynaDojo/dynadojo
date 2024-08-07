from dynadojo.wrappers import SystemChecker
import numpy as np
import pandas as pd
import json
import os
import dysts
from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
from dynadojo.utils.complexity_measures import gp_dim, mse_mv, pca, find_lyapunov_exponents, find_max_lyapunov, kaplan_yorke_dimension, pesin

base_path = os.path.dirname(dysts.__file__)
json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')
with open(json_file_path, 'r') as file:
    systems_data = json.load(file)
all_systems = list(systems_data.keys())

problematic_systems = ["IkedaDelay", "MackeyGlass", "PiecewiseCircuit", "RabinovichFabrikant", 
                       "ScrollDelay", "SprottDelay", "SprottJerk", "SprottL", "SprottM", "SprottQ", "VossDelay", "Torus", "TurchinHanski"]

for problematic_system in problematic_systems:
    all_systems.remove(problematic_system)

"""For all mathematical properties we perform 20 replicate computations from different initial conditions, and record
the average in our database. To ensure high-quality estimates, we compute trajectories at high granularity of 500
points per period (as determined by the dominant frequency in the power spectrum), and we use trajectories with
length 2500, corresponding to five complete periods."""

timesteps = 2500
pts_per_period = 500

# prep dataframe columns
column_names = ["system", "calc_correlation_dimension", "calc_multiscale_entropy", "calc_lyapunov_spectrum_estimated", "calc_pesin_entropy", "calc_kaplan_yorke_dimension",
                "og_correlation_dimension", "og_multiscale_entropy", "og_lyapunov_spectrum_estimated", "og_pesin_entropy", "og_kaplan_yorke_dimension",
                "diff_correlation_dimension", "diff_multiscale_entropy", "diff_lyapunov_spectrum_estimated", "diff_pesin_entropy", "diff_kaplan_yorke_dimension"
                ]

# Checkpointing system to reduce redundancy if partial data already exists
file_path = 'docs/complexity_comparison.JSON'
pd.DataFrame(columns=column_names).to_json(file_path, orient='records', lines=True)
df = pd.read_json(file_path, orient='records', lines=True) # loads newly created data file

for system_name in all_systems:
    system_dict = systems_data[system_name]
    embedding_dimension = system_dict["embedding_dimension"]
    initial_conditions = np.asarray([system_dict["initial_conditions"]])

    system = SystemChecker(GilpinFlowsSystem(latent_dim=embedding_dimension, embed_dim=embedding_dimension, system_name=system_name))
    model = system._system.system
    
    tpts, x = system._system.make_data(initial_conditions, pts_per_period=pts_per_period, timesteps=timesteps, return_times=True)
    x = x[0]

    lyapunov_spectrum = find_lyapunov_exponents(x, tpts, timesteps, model)
    
    # Values calculated with Dynadojo utils
    calc_correlation_dimension = gp_dim(x)
    calc_multiscale_entropy = mse_mv(x)
    calc_lyapunov_spectrum_estimated = pca(x)
    # calc_maximum_lyapunov_estimated = find_max_lyapunov(lyapunov_spectrum)
    calc_pesin_entropy = kaplan_yorke_dimension(lyapunov_spectrum)
    calc_kaplan_yorke_dimension = pesin(lyapunov_spectrum)

    # Values pulled from Gilpin JSON
    og_correlation_dimension = system_dict.get("correlation_dimension")
    og_multiscale_entropy = system_dict.get("multiscale_entropy")
    og_lyapunov_spectrum_estimated = system_dict.get("lyapunov_spectrum_estimated")
    # og_maximum_lyapunov_estimated = system_dict.get("maximum_lyapunov_estimated")
    og_pesin_entropy = system_dict.get("pesin_entropy")
    og_kaplan_yorke_dimension = system_dict.get("kaplan_yorke_dimension")

    # Difference between values
    if og_correlation_dimension != None:
        diff_correlation_dimension = calc_correlation_dimension - og_correlation_dimension
    else:
        diff_correlation_dimension = None
    if og_multiscale_entropy != None:
        diff_multiscale_entropy = calc_multiscale_entropy - og_multiscale_entropy
    else:
        diff_multiscale_entropy = None
    if og_lyapunov_spectrum_estimated != None:
        diff_lyapunov_spectrum_estimated = calc_lyapunov_spectrum_estimated - og_lyapunov_spectrum_estimated
    else:
        og_lyapunov_spectrum_estimated = None
    # diff_maximum_lyapunov_estimated = calc_maximum_lyapunov_estimated - og_maximum_lyapunov_estimated
    if og_pesin_entropy != None:
        diff_pesin_entropy = calc_pesin_entropy - og_pesin_entropy
    else:
        diff_pesin_entropy = None
    if og_kaplan_yorke_dimension != None:
        diff_kaplan_yorke_dimension = calc_kaplan_yorke_dimension - og_kaplan_yorke_dimension
    else:
        diff_kaplan_yorke_dimension = None
    
    # Throw the values into a list and then into pandas dataframe
    data = [[system_name, calc_correlation_dimension, calc_multiscale_entropy, calc_lyapunov_spectrum_estimated, calc_pesin_entropy, calc_kaplan_yorke_dimension,
                og_correlation_dimension, og_multiscale_entropy, og_lyapunov_spectrum_estimated, og_pesin_entropy, og_kaplan_yorke_dimension,
                diff_correlation_dimension, diff_multiscale_entropy, diff_lyapunov_spectrum_estimated, diff_pesin_entropy, diff_kaplan_yorke_dimension
            ]]
    temp_df = pd.DataFrame(data, columns=column_names)
    temp_df.to_json(file_path, mode='a', orient='records', lines=True)

df_unsorted = pd.read_json(file_path, orient='records', lines=True)
df = df_unsorted.sort_values(by=['system', 'seed', 'D'])
df.to_json(file_path, orient='records', lines=True)
    


