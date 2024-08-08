import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('/home/vigith/dynadojo/docs/dyna_complexity_data.JSON', orient='records', lines=True)
OOD = [True, False]
timesteps_list = [100, 500, 1000, 2500, 5000, 10000]
#dimensions = [4, 8, 12]
dimensions = [3, 5, 7]
all_systems = ["LorenzSystem", "KuramotoSystem", "CompetitiveLVSystem", "PreyPredatorSystem", "NBodySystem"]
bin_edges = []
for i in range(10):
    bin_edges.append(0.05*i)
for timesteps in timesteps_list:
    filter = df['timesteps'] == timesteps
    filtered_df = df[filter]
    plt.hist(filtered_df['mse_mv'], bins=bin_edges, edgecolor='black')
    plt.title(f"Histogram of mse_mv at timesteps {timesteps}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 20)
    plt.savefig(f"mse_mv_timesteps_{timesteps}.png")
    plt.clf()
bin_edges = []
for i in range(10):
    bin_edges.append(0.1*i)
for dimension in dimensions:
    filter = df['D'] == dimension
    filtered_df = df[filter]
    plt.hist(filtered_df['mse_mv'], bins=bin_edges, edgecolor='black')
    plt.title(f"Histogram of mse_mv at dimension {dimension}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 40)
    plt.savefig(f"mse_mv_dimension_{dimension}.png")
    plt.clf()
bin_edges = []
for i in range(10):
    bin_edges.append(0.1*i)
for state in OOD:
    filter = df['OOD'] == state
    filtered_df = df[filter]
    plt.hist(filtered_df['mse_mv'], bins=10, edgecolor='black')
    plt.title(f"Histogram of mse_mv if OOD {state}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 60)
    plt.savefig(f"mse_mv_OOD_{state}.png")
    plt.clf()