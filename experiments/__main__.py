"""
Command line interface for running experiments.
Arguments for make:
    --algo: which algo, short name, see params.py algo_dict
    --system: which system, short name, see params.py system_dict
    --challenge: which challenge, one of ["fc", "fts", "fe"]
    --output_dir: where to save config, default "experiments/outputs"
    --all: if True, make all params, default False
Usage:
    python -m experiments make --challenge <challenge_key> --system <system_key> --algo <algo_key> --output_dir <output_dir>
    python -m experiments make --challenge fc --system lds --algo lr_5

Arguments for run:
    --config_file: which config file to run
    --total_nodes: how many machines to run on (default 1, for running locally)
    --node: which node is being run, [1, total_nodes], default None which runs the whole challenge
    --output_dir: where to save results, default "experiments/outputs"
    --num_cpu_parallel: number of cpus to use for parallelization, default None which runs without parallelization
    --jobs: which jobs to run, comma separated list of integers, default None which runs all jobs
    --if_missing: if True, only run missing jobs, default False
Usage:
    python -m experiments \
        run \
        --config_file experiments/outputs/fc/lds/fc_lds_lr_l=10/config.json \
        --node 2 --total_nodes 10 \
        --num_cpu_parallel -2 \
        --if_missing

    python -m experiments run --num_cpu_parallel -2 --config_file experiments/outputs/fc/lds/fc_lds_lr_5_l=5/config.json 
    
Arguments for plot:
    --data_dir: where to load results from
    --output_dir: where to save plots, default "experiments/outputs"

Usage:
    python -m experiments plot --data_dir experiments/outputs/fc/lds/fc_lds_lr_l=10 --output_dir experiments/outputs

Arguments for check:
    --data_dir: where to load results from

Usage:
    python -m experiments check --data_dir experiments/outputs/fc/lds/fc_lds_lr_l=10

Usage:
    (see dynadojo.sbatch)
    python -m experiments --challenge fc --algo lr --system lds
    python -m experiments --challenge fts --algo lr --system lds
    python -m experiments --challenge fe --algo lr --system lds

python -m experiments make --challenge fe --algo lr --system lds --output_dir="experiments/outputs/scratch"
"""

import argparse
import os
import json
from .utils import algo_dict, load_from_json, system_dict, challenge_dicts
from .main import load_data, run_challenge, make_plots, save_config, green, pink, cyan, red, loadingBar, bold
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize


program = argparse.ArgumentParser(description='DynaDojo Experiment CLI')
subparsers = program.add_subparsers(dest='command', help='sub-command help')
make_parser = subparsers.add_parser('make', help='Generate an experiement param file')
run_parser = subparsers.add_parser('run', help='Run an experiment param file')
plot_parser = subparsers.add_parser('plot', help='Plot an experiment results')
check_parser = subparsers.add_parser('check', help='Check for missing jobs')
scale_parser = subparsers.add_parser('scale', help='Temporary utility which rescales losses by dimensionality')
status_parser =subparsers.add_parser('status', help='List all available config.json files that you have already made')

# Accept command line arguments
make_parser.add_argument('--algo', type=str, default='lr', help='Specify which algo to run')
make_parser.add_argument('--system', type=str, default='lds', choices=system_dict.keys(), help='Specify which system to run')
make_parser.add_argument('--make_status', action='store_true', help='list all the available experiments you can call the "make" command on')
status_parser.set_defaults(make_status=True)

make_parser.add_argument('--challenge', type=str, default="fc", choices=["fc", "fts", "fe"], help='Specify which challenge to run')
make_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save config')
make_parser.add_argument('--all', action='store_true', help='if True, make all params')
make_parser.set_defaults(all=False)

run_parser.add_argument('--config_file', type=str, help='what config file to run')
run_parser.add_argument('--node', type=int, default=None, help='which node is being run in [1, total_nodes], if None, run on splits')
run_parser.add_argument('--total_nodes', type=int, default=1, help='how many machines to run on')
run_parser.add_argument('--num_cpu_parallel', type=int, default=None, help='number of cpus to use for parallelization')
run_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save results')
run_parser.add_argument('--jobs', type=str, default=None, help='which jobs to run, comma separated list of integers')
run_parser.add_argument('--if_missing', action='store_true', help='if True, only run missing jobs')
run_parser.set_defaults(if_missing=False)

plot_parser.add_argument('--data_dir', type=str, help='where to load results from')
plot_parser.add_argument('--output_dir', type=str, default="experiments/outputs", help='where to save plots')

check_parser.add_argument('--data_dir', type=str, help='where to load results from')

scale_parser.add_argument('--data_dir', type=str, help='where to load results from')

# status_parser.add_argument('--system', type=str, default=None, choices=system_dict.keys(), help='filter by system')
status_parser.add_argument('--is_complete', type=str, choices=['true','false'],help='filter by completed experiments')
status_parser.add_argument('--algo', type=str, help='Specify which algo to filter through')
status_parser.add_argument('--system', type=str, help='Specify which system to filter through')
status_parser.add_argument('--challenge', type=str, choices=["fc", "fts", "fe"], help='Specify which challenge to filter through')
status_parser.set_defaults(make=False)

args, rest = program.parse_known_args()

if args.command == 'make':
    if args.make_status:
        for challenge in challenge_dicts.keys():
            for system in system_dict.keys():
                for algo in algo_dict.keys():
                    print('--challenge =',challenge,'--system =', system, '--algo =',algo)
    else:     
        if args.all:
            for c, chall_dict in challenge_dicts.values():
                for s in chall_dict.keys():
                    if s != "default":
                        for a in chall_dict[s].keys():
                            if a != "default":
                                print(f"Making {c.__name__} {s} {a}")
                                config_file, total_jobs = save_config(s, a, challenge_cls=c, output_dir=args.output_dir)
                                print(pink(f"{config_file} with {total_jobs} jobs"))
        else:
            assert args.algo.split("_")[0] in algo_dict.keys(), f"algo {args.algo} must be in algo_dict"
            if args.challenge == "fc":
                challenge_cls = FixedComplexity
            elif args.challenge == "fts":
                challenge_cls = FixedTrainSize
            else:
                challenge_cls = FixedError
            config_file, total_jobs = save_config(args.system, args.algo, challenge_cls, output_dir=args.output_dir)
            print(pink(f"{config_file} with {total_jobs} jobs"))
            if rest: #maybe parse more args
                args = program.parse_args(rest) 
                if args.command == 'run':
                    args.config_file = config_file
                else:
                    exit(0)

if args.command == 'run':
    assert args.config_file is not None, "must specify config file"
    
    if args.if_missing:
        config = load_from_json(args.config_file)
        total_jobs = config["total_jobs"]
        _, data = load_data(os.path.join(args.output_dir, config["folder_path"]))
        if data is None:
            print(green("No previous jobs found."))
            args.jobs = None
        else:
            completed_jobs = data['job_id'].drop_duplicates().to_list()
            missing_jobs = [i for i in range(total_jobs) if i not in completed_jobs]
            if len(missing_jobs) == 0:
                print(green("All jobs already completed. Exiting."))
                exit(0)
            print(green(f"{len(missing_jobs)} missing jobs found. Only running missing jobs."))
            args.jobs = ','.join(map(str, missing_jobs))
        
    if args.node is not None and args.total_nodes > 1:
        assert args.node >= 1 and args.node <= args.total_nodes, f"{args.node=} must be between[1, {args.total_nodes=}]"
        run_challenge(
            config_file_path=args.config_file,
            output_dir=args.output_dir,
            split=(args.node, args.total_nodes),
            num_cpu_parallel=args.num_cpu_parallel,
            jobs_filter=[int(j) for j in args.jobs.split(",")] if args.jobs else None
        )
    else: # run the whole challenge
        print(green(f"Running {len(args.jobs.split(',')) if args.jobs else 'all'} jobs."))
        run_challenge(
            config_file_path=args.config_file,
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
    #must contain config.json
    assert args.data_dir is not None, "must specify data directory"
    assert os.path.exists(args.data_dir + "/config.json"), f"config.json not found in {args.data_dir}"

    config = load_from_json(args.data_dir + "/config.json")
    total_jobs = config["total_jobs"]
    _, data = load_data(args.data_dir)
    if data is None:
        completed_jobs = []
    else:
        completed_jobs = data['job_id'].drop_duplicates().to_list()
    missing_jobs = [i for i in range(total_jobs) if i not in completed_jobs]
    if len(missing_jobs) == 0:
        print(green("All jobs completed."))
        exit(0)
    print(f"Num of missing jobs: \t {len(missing_jobs)} of {total_jobs}")
    print(f"Missing jobs: \n{','.join(map(str, missing_jobs))}")

elif args.command == 'scale': 
    assert args.data_dir is not None, "must specify data directory"
    files, data = load_data(args.data_dir)
    # make a new subfolder for old data inside data_dir
    data_dir_unscaled = args.data_dir + "/original_data"
    try:
        os.makedirs(data_dir_unscaled, exist_ok=False)
    except FileExistsError:
        print(pink(f"Exiting...Already scaled data. {data_dir_unscaled} already exists. "))
        exit(0)

    # move all csv files in data_dir to data_dir_unscaled
    for filepath in files:
        os.rename(filepath, data_dir_unscaled + "/" + os.path.basename(filepath))
    print(green(f"Original data moved to {data_dir_unscaled}"))
    
    # rescale all losses by dimensionality
    data['error'] = data['error'] * data['latent_dim']
    data['ood_error'] = data['ood_error'] * data['latent_dim']

    # save the new data as csv file in data_dir
    data.to_csv(args.data_dir + "/data.csv", index=False)
    print(green(f"Rescaled data saved to {args.data_dir}/data.csv"))

elif args.command == 'status':
    experiment_list = []  # all the config.json files in the outputs folder
    experiment_dict = {}
    algo_filter = args.algo
    system_filter = args.system
    challenge_filter = args.challenge
    complete_filter = args.is_complete

    directory_path = 'experiments/outputs'

    # Find all 'config.json' files, add filepath to a list, sorted by challenge type
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for file in filenames:
            if file.endswith('config.json'):
                file_path = os.path.join(dirpath, file)
                f = open(file_path, 'r')
                experiment = json.load(f)
                
                algo_cls = experiment.get('evaluate', {}).get('algo_cls', {})
                algo_cls_name = algo_cls.get('class_name', '')
                system_cls = experiment.get('challenge', {}).get('system_cls', {})
                system_cls_name = system_cls.get('class_name', '')
                challenge_cls = experiment.get('challenge_cls', {})
                challenge_cls_name = challenge_cls.get('class_name', '')

                # Check algorithm if filter is set
                if algo_filter:
                    algo_filter_name = {
                        'lr': 'LinearRegression',
                        'dnn': 'DNN',
                        'sindy': 'SINDy'
                    }.get(algo_filter, '')
                    if algo_cls_name != algo_filter_name:
                        continue
                
                # Check system if filter is set
                if system_filter:
                    system_filter_name = {
                        'lds': 'LDSystem',
                        'lorenz': 'LorenzSystem',
                        'lv_p': 'PreyPredatorSystem',
                        'epi_1': 'LorenzSystem',
                        'nbody': 'LorenzSystem',
                        'heat': 'LorenzSystem',
                        'fbsnn_1': 'LorenzSystem',
                        'fbsnn_2': 'LorenzSystem',
                        'ctln': 'LorenzSystem',
                        'kura': 'LorenzSystem'
                    }.get(system_filter, '')
                    if system_cls_name != system_filter_name:
                        continue

                # Check challenge if filter is set
                if challenge_filter:
                    challenge_filter_name = {
                        'fc': 'FixedComplexity',
                        'fts': 'FixedTrainSize',
                        'fe': 'FixedError'
                    }.get(challenge_filter, '')
                    if challenge_cls_name != challenge_filter_name:
                        continue
                
                
                experiment_type = experiment['challenge_cls']['class_name']
                total_jobs = experiment.get("total_jobs", 0)
                _, data = load_data(dirpath, print_status=False)
                if data is None:
                    completed_jobs = []
                else:
                    completed_jobs = data['job_id'].drop_duplicates().to_list()

                complete_cls = str(len(completed_jobs) == total_jobs).lower()
                
                #Check complete if filter is set 
                if complete_filter:
                    if complete_cls != complete_filter:
                        continue
                
                # Sort
                if experiment_type in experiment_dict.keys():
                    experiment_dict[experiment_type].append({
                        'total_jobs': total_jobs,
                        'complete_jobs': len(completed_jobs),
                        'folder_path': file_path
                    })
                else:
                    experiment_dict[experiment['challenge_cls']['class_name']] = [{
                        'total_jobs': total_jobs,
                        'complete_jobs': len(completed_jobs),
                        'folder_path': file_path
                    }]

    # Check if experiments made
    if not experiment_dict:
        print(cyan(bold('No experiments made')))
        print(bold('Experiment configs available: 0'), end=' ')
        print(loadingBar(0, 1, 40))

        # Hints to make experiments
        print('\033[1;31m'+'To make an experiment:'+'\033[0m')
        print('\033[0;31m'+'    python -m experiments make'+'\033[0m')

    else:
        # Determine max length for formatting, and count total jobs
        max_length = 0
        max_length_job = 0
        all_jobs = 0
        all_finished_jobs = 0
        job_dict = {}

        # Get max length for formatting
        # Get all jobs for each experiment type (for progress bar)
        for challenge_type in experiment_dict.keys():
            output_list = [path for path in experiment_dict[challenge_type]]

            # Format job numbers red or green depending on status
            max_length = max(max_length, max(len(' ' + path['folder_path']) for path in output_list))
            max_length_job = max(max_length_job, max(len(f"{path['complete_jobs']} / {path['total_jobs']} Jobs") for path in output_list))

            all_jobs += sum(jobs['total_jobs'] for jobs in experiment_dict[challenge_type])
            all_finished_jobs += sum(jobs['complete_jobs'] for jobs in experiment_dict[challenge_type])
            job_dict[challenge_type] = {'all_jobs': all_jobs, 'all_completed_jobs': all_finished_jobs}

        max_title = max(len(challenge_type) for challenge_type in experiment_dict.keys())

        # Print Title
        print(bold(f'Experiment configs available: {all_jobs}'), end=' ')
        print(loadingBar(all_finished_jobs, all_jobs, max_length + max_length_job - len(f'Experiment configs available: {all_jobs}') + 9))
        print('\033[1;31m'+'To run an experiment:'+'\033[0m')
        print('\033[0;31m'+'    python -m experiments run --config_file <name>\n'+'\033[0m')

        # Print paths by Challenge Type
        for challenge_type in experiment_dict.keys():
            print(bold(f"{challenge_type}: " + ' ' * (max_title - len(challenge_type)) + str(len(experiment_dict[challenge_type]))), end=' ')
            print(loadingBar(job_dict[challenge_type]['all_completed_jobs'], job_dict[challenge_type]['all_jobs'], 20))

            output_list = [path for path in experiment_dict[challenge_type]]

            # Print paths
            for path in output_list:
                output = path['folder_path']

                # Bolding experiment part of the filepath
                output_bold = str(output).split('/')
                output_bold[-2] = bold(output_bold[-2], color='\033[96m')
                output_str = ''
                for out in output_bold:
                    output_str += out + '/'
                output_str = cyan(output_str[0:-1])

                print('    '+output_str+' '*((max_length-len(output)+(max_length_job-len(str(path['complete_jobs'])+' / '+str(path['total_jobs'])+' Jobs')))), end ='')

                # Print number of jobs + progress bar
                if path['complete_jobs'] == path['total_jobs']:
                    print(green(f'{path["complete_jobs"]}'), '/', green(f'{path["total_jobs"]}'), end=' ')
                else:
                    print(red(f'{path["complete_jobs"]}'), '/', red(f'{path["total_jobs"]}'), end=' ')

                print(loadingBar(path['complete_jobs'], path['total_jobs'], 10))
            print()