"""
Command line interface for running experiments.
Arguments for make:
    --algo: which algo, short name, see params.py algo_dict
    --system: which system, short name, see params.py system_dict
    --challenge: which challenge, one of ["fc", "fts", "fe"]
    --output_dir: where to save params
Usage:
    python -m experiments make --challenge fc --algo lr --system lds

Arguments for run:
    --params_file: which params file to run
    --total_nodes: how many machines to run on (default 1, for running locally)
    --node: which node is being run, [1, total_nodes], default None which runs the whole challenge
    --output_dir: where to save results
    --num_cpu_parallel: number of cpus to use for parallelization, default None which runs without parallelization
Usage:
    python -m experiments \
        run \
        --params_file experiments/outputs/fc/lds/fc_lds_lr_l=10/params.json \
        --node 2 --total_nodes 10 \
        --num_cpu_parallel -5
    
Arguments for plot:
    --data_dir: where to load results from
    --output_dir: where to save plots

Usage:

Usage:
    (see dynadojo.sbatch)
    python -m experiments --challenge fc --algo lr --system lds
    python -m experiments --challenge fts --algo lr --system lds
    python -m experiments --challenge fe --algo lr --system lds

python -m experiments make --challenge fe --algo lr --system lds --output_dir="experiments/outputs/scratch"
"""

import argparse
from .params import algo_dict, load_from_json, system_dict
from .main import load_data, run_challenge, make_plots, get_max_splits, save_params, prGreen
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize


program = argparse.ArgumentParser(description='DynaDojo Experiment CLI')
subparsers = program.add_subparsers(dest='command', help='sub-command help')
make_parser = subparsers.add_parser('make', help='Generate an experiement param file')
run_parser = subparsers.add_parser('run', help='Run an experiment param file')
plot_parser = subparsers.add_parser('plot', help='Plot an experiment results')
check_parser = subparsers.add_parser('check', help='Check for missing jobs')

# # Accept command line arguments
make_parser.add_argument('--algo', type=str, default='lr', help='Specify which algo to run')
make_parser.add_argument('--system', type=str, default='lds', choices=system_dict.keys(), help='Specify which system to run')
make_parser.add_argument('--challenge', type=str, default="fc", choices=["fc", "fts", "fe"], help='Specify which challenge to run')
make_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save params')

run_parser.add_argument('--params_file', type=str, help='what params file to run')
run_parser.add_argument('--node', type=int, default=None, help='which node is being run in [1, total_nodes], if None, run on splits')
run_parser.add_argument('--total_nodes', type=int, default=1, help='how many machines to run on')
run_parser.add_argument('--num_cpu_parallel', type=int, default=None, help='number of cpus to use for parallelization')
run_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save results')
run_parser.add_argument('--jobs', type=str, default=None, help='which jobs to run, comma separated list of integers')
run_parser.add_argument('--check_dir', type=str, default=None, help='Data dir path in order to check and run only missing jobs')

plot_parser.add_argument('--data_dir', type=str, help='where to load results from')
plot_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save plots')

check_parser.add_argument('--data_dir', type=str, help='where to load results from')

args = program.parse_args()

if args.command == 'make':
    assert args.algo.split("_")[0] in algo_dict.keys(), f"algo {args.algo} must be in algo_dict"
    if args.challenge == "fc":
        challenge_cls = FixedComplexity
    elif args.challenge == "fts":
        challenge_cls = FixedTrainSize
    else:
        challenge_cls = FixedError
    params_file = save_params(args.system, args.algo, challenge_cls, output_dir=args.output_dir)
    print(params_file)

elif args.command == 'run':
    assert args.params_file is not None, "must specify params file"
    
    if args.check_dir is not None:
        params = load_from_json(args.check_dir + "/params.json")
        total_jobs = params["total_jobs"]
        _, data = load_data(args.check_dir)
        completed_jobs = data['job_id'].drop_duplicates().to_list()
        missing_jobs = [i for i in range(total_jobs) if i not in completed_jobs]
        if len(missing_jobs) == 0:
            prGreen("All jobs already completed. Exiting.")
            exit(0)
        prGreen(f"{len(missing_jobs)} missing jobs found. Only running missing jobs.")
        args.jobs = ','.join(map(str, missing_jobs))
        
    if args.node is not None and args.total_nodes > 1:
        assert args.node >= 1 and args.node <= args.total_nodes, f"{args.node=} must be between[1, {args.total_nodes=}]"
        run_challenge(
            params_file_path=args.params_file,
            output_dir=args.output_dir,
            split=(args.node, args.total_nodes),
            num_cpu_parallel=args.num_cpu_parallel,
            jobs_filter=[int(j) for j in args.jobs.split(",")] if args.jobs else None
        )
    else: # run the whole challenge
        prGreen("Running the whole challenge.")
        run_challenge(
            params_file_path=args.params_file,
            output_dir=args.output_dir,
            split=None,
            num_cpu_parallel=args.num_cpu_parallel,
            jobs_filter=[int(j) for j in args.jobs.split(",")] if args.jobs else None
        )

elif args.command == 'plot':
    assert args.data_dir is not None, "must specify data directory"
    make_plots(
        data_path=args.data_dir,
        output_dir=args.output_dir,
        save=True
    )

elif args.command == 'check':
    params = load_from_json(args.data_dir + "/params.json")
    total_jobs = params["total_jobs"]
    _, data = load_data(args.data_dir)
    completed_jobs = data['job_id'].drop_duplicates().to_list()
    missing_jobs = [i for i in range(total_jobs) if i not in completed_jobs]
    if len(missing_jobs) == 0:
        prGreen("All jobs completed.")
        exit(0)
    print(f"Num of missing jobs: \t {len(missing_jobs)}")
    print(f"Missing jobs: \n{','.join(map(str, missing_jobs))}")

    


