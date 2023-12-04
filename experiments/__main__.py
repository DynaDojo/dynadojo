"""
Command line interface for running experiments.
Arguments:
    --algo: which algo to run, short name, see params.py algo_dict
    --system: which system to run, short name, see params.py system_dict
    --challenge: which challenge to run, one of ["fc", "fts", "fe"]
    --total_nodes: how many machines to run on (default 1, for running locally)
    --node: which node is being run, [1, total_nodes], default None which runs the whole challenge
    --output_dir: where to save outputs
    --plot: whether to plot the results OR to run the challenge
Usage:
    (see dynadojo.sbatch)
    python -m experiments --challenge fc --algo lr --system lds
    python -m experiments --challenge fts --algo lr --system lds
    python -m experiments --challenge fe --algo lr --system lds

python -m experiments --challenge fe --algo lr --system lds --plot --output_dir="experiments/outputs/scratch"
"""

import argparse
from .params import algo_dict, system_dict
from .main import run_challenge, make_plots, get_max_splits
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize

# # Accept command line arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--algo', type=str, default='lr', help='Specify which algo to run')
parser.add_argument('--system', type=str, default='lds', choices=system_dict.keys(), help='Specify which system to run')
parser.add_argument('--challenge', type=str, default="fc", choices=["fc", "fts", "fe"], help='Specify which challenge to run')
parser.add_argument('--total_nodes', type=int, default=1, help='how many machines to run on')
parser.add_argument('--node', type=int, default=None, help='which node is being run in [1, total_nodes], if None, run on splits')
parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save outputs')
parser.add_argument('--plot', default=False, action='store_true', help='whether to plot results')
parser.add_argument('--num_cpu_parallel', type=int, default=None, help='number of cpus to use for parallelization')
# add argument for test id in [1,2,3]
args = parser.parse_args()

assert args.algo.split("_")[0] in algo_dict.keys(), f"algo {args.algo} not in algo_dict"

if args.challenge == "fc":
    challenge_cls = FixedComplexity
elif args.challenge == "fts":
    challenge_cls = FixedTrainSize
else:
    challenge_cls = FixedError



if args.plot:
    make_plots(
        s = args.system,
        a = args.algo,
        output_dir=args.output_dir,
        challenge_cls = challenge_cls)
else:
    # if node is specified, run that split
    if args.node is not None and args.total_nodes > 1:
        assert args.node >= 1 and args.node <= args.total_nodes, "node must be in [1, total_nodes]"
        max_splits = min(args.total_nodes, get_max_splits(s = args.system, m = args.algo, challenge_cls = challenge_cls))


        if args.node <= max_splits:
            run_challenge(
                s =args.system,
                a = args.algo,
                output_dir=args.output_dir, 
                split=(args.node, max_splits),
                challenge_cls=challenge_cls,
                num_cpu_parallel=args.num_cpu_parallel,
        )
    else: # run the whole challenge
        run_challenge(
            s = args.system,
            a = args.algo,
            output_dir=args.output_dir, 
            challenge_cls=challenge_cls,
            num_cpu_parallel=args.num_cpu_parallel,
        )
    

