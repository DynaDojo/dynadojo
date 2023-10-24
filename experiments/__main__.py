import argparse
import dynadojo as dd
from .experiments import run_fc, make_plots, get_max_splits
from dynadojo.challenges import  FixedError, FixedComplexity, FixedTrainSize

# # Accept command line arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--exp', type=int, default=1, help='0=test, 1=experiment')
parser.add_argument('--test', type=int, default=0, help='0=FixedComplexity, 1=FixedTrain, 2=FixedError')
parser.add_argument('--nodes', type=int, default=20, help='how many machines to run on')
# add argument for test id in [1,2,3]
args = parser.parse_args()

if args.exp == 1:

    max_splits = min(args.nodes, get_max_splits(s ="lds", m = "lr", challenge_cls = FixedComplexity))

    for split in range(max_splits):
        run_fc(
            s ="lds",
            m = "lr",
            output_dir="experiments/outputs", 
            split=(split+1, max_splits)
        )
    make_plots(
        s ="lds",
        m = "lr",
        output_dir="experiments/outputs",
        challenge_cls = FixedComplexity)
else:
    # dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, reps_filter=None, test_ids=[0, 1,2])
    dd.testing.test_system("dynadojo.systems.lds", seed = 2000, reps=20, reps_filter=[], L_filter=[], test_ids=[args.test],
        plot = True,
        # model_kwargs={"seed" : 160033707},
        # system_kwargs={"seed" : 2144784513},
    )
    # dd.testing.test_system("dynadojo.systems.lds", L= [20, 100], n_starts=[100,100], t=10, seed = 10, reps=2, reps_filter=[0], L_filter=[20], test_ids=[2])

    # Should not run anything because of L_filter out of range
    # dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, reps_filter=[0], L_filter=[5] , test_ids=[2])

