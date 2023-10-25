"""
Command line interface for running experiments.
Arguments:
    --challenge: which challenge to run, one of ["fc", "fts", "fe"]
    --nodes: how many machines to run on
    --node_id: which node is being run in [0, nodes-1], if None, run on splits
Usage:
    python -m experiments --challenge fc --nodes 20
    python -m experiments --challenge fts --nodes 20
    python -m experiments --challenge fe --nodes 20
"""

import argparse
from .main import run_challenge, make_plots, get_max_splits
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize

# # Accept command line arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--challenge', type=str, default="fc", choices=["fc", "fts", "fe"], help='Specify which challenge to run')
parser.add_argument('--nodes', type=int, default=20, help='how many machines to run on')
parser.add_argument('--node_id', type=int, default=None, help='which node is being run in [0, nodes-1], if None, run on splits')
parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save outputs')
# add argument for test id in [1,2,3]
args = parser.parse_args()



if args.challenge == "fc":
    challenge_cls = FixedComplexity
elif args.challenge == "fts":
    challenge_cls = FixedTrainSize
else:
    challenge_cls = FixedError

max_splits = min(args.nodes, get_max_splits(s ="lds", m = "lr", challenge_cls = challenge_cls))

for split in range(max_splits):
    run_challenge(
        s ="lds",
        m = "lr",
        output_dir=args.output_dir, 
        split=(split+1, max_splits),
        challenge_cls=challenge_cls,
    )
make_plots(
    s ="lds",
    m = "lr",
    output_dir=args.output_dir,
    challenge_cls = challenge_cls)
    

