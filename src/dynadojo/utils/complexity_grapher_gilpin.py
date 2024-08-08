import pandas as pd
import matplotlib.pyplot as plt

import os
import dysts
import json

df = pd.read_json('/home/vigith/dynadojo/docs/dyna_complexity_data.JSON', orient='records', lines=True)
OOD = [True, False]
timesteps_list = [100, 500, 1000, 2500, 5000, 10000]
#dimensions = [4, 8, 12]
dimensions = [3]#, 5, 7]

base_path = os.path.dirname(dysts.__file__)
json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')
with open(json_file_path, 'r') as file:
    systems_data = json.load(file)
all_systems = list(systems_data.keys())

problematic_systems = ["IkedaDelay", "MackeyGlass", "PiecewiseCircuit", "ScrollDelay", "SprottDelay", "SprottJerk", "VossDelay", "Torus"]
for problematic_system in problematic_systems:
    all_systems.remove(problematic_system)

measures = ["gp_dim", "mse_mv", "pca", "lyapunov_spectrum", "kaplan_yorke_dimension", "pesin"]
for measure in measures:    
    #Sweep Timesteps
    bin_edges = []
    for i in range(10):
        bin_edges.append(0.05*i)
    for timesteps in timesteps_list:
        filter = df['timesteps'] == timesteps
        filtered_df = df[filter]
        plt.hist(filtered_df[measure], bins=bin_edges, edgecolor='black')
        plt.title(f"Histogram of {measure} at timesteps {timesteps}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        #plt.ylim(0, 20)
        plt.savefig(f"{measure}_timesteps_{timesteps}.png")
        plt.clf()

    #Sweep Dimension
    bin_edges = []
    for i in range(10):
        bin_edges.append(0.1*i)
    for dimension in dimensions:
        filter = df['D'] == dimension
        filtered_df = df[filter]
        plt.hist(filtered_df[measure], bins=bin_edges, edgecolor='black')
        plt.title(f"Histogram of {measure} at dimension {dimension}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        #plt.ylim(0, 40)
        plt.savefig(f"{measure}_dimension_{dimension}.png")
        plt.clf()

    #Sweep In/Out
    bin_edges = []
    for i in range(10):
        bin_edges.append(0.1*i)
    for state in OOD:
        filter = df['OOD'] == state
        filtered_df = df[filter]
        plt.hist(filtered_df[measure], bins=10, edgecolor='black')
        plt.title(f"Histogram of {measure} if OOD {state}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        #plt.ylim(0, 60)
        plt.savefig(f"{measure}_OOD_{state}.png")
        plt.clf()