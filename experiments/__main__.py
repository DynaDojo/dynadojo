"""
Command line interface for running experiments.
Arguments:
    --challenge: which challenge to run, one of ["fc", "fts", "fe"]
    --total_nodes: how many machines to run on
    --node: which node is being run, [0, nodes-1], default None which runs the whole challenge
Usage:
    python -m experiments --challenge fc --nodes 20
    python -m experiments --challenge fts --nodes 20
    python -m experiments --challenge fe --nodes 20
"""

import argparse
from .params import model_dict, system_dict
from .main import run_challenge, make_plots, get_max_splits
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize

# # Accept command line arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--model', type=str, default='lr', choice=model_dict.keys(), help='Specify which model to run')
parser.add_argument('--system', type=str, default='lds', choice=system_dict.keys(), help='Specify which system to run')
parser.add_argument('--challenge', type=str, default="fc", choices=["fc", "fts", "fe"], help='Specify which challenge to run')
parser.add_argument('--total_nodes', type=int, default=1, help='how many machines to run on')
parser.add_argument('--node', type=int, default=None, help='which node is being run in [0, nodes-1], if None, run on splits')
parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save outputs')
parser.add_argument('--plot', default=False, action='store_true', help='whether to plot results')
# add argument for test id in [1,2,3]
args = parser.parse_args()



if args.challenge == "fc":
    challenge_cls = FixedComplexity
elif args.challenge == "fts":
    challenge_cls = FixedTrainSize
else:
    challenge_cls = FixedError



if args.plot:
    make_plots(
        s = args.system,
        m = args.model,
        output_dir=args.output_dir,
        challenge_cls = challenge_cls)
else:
    # if node is specified, run that split
    if args.node is not None and args.total_nodes > 1:
        assert args.node >= 0 and args.node < args.total_nodes, "node must be in [0, nodes-1]"
        max_splits = min(args.total_nodes, get_max_splits(s = args.system, m = args.model, challenge_cls = challenge_cls))

        if args.node < max_splits:
            run_challenge(
                s =args.system,
                m = args.model,
                output_dir=args.output_dir, 
                split=(args.node+1, max_splits),
                challenge_cls=challenge_cls,
        )
    else: # run the whole challenge
        run_challenge(
            s = args.system,
            m = args.model,
            output_dir=args.output_dir, 
            challenge_cls=challenge_cls,
        )
    

