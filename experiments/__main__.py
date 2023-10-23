import argparse
import dynadojo as dd
print("hello")
# dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, eval_reps=None, test_ids=[0, 1,2])
dd.testing.test_system("dynadojo.systems.lds", seed = 2000, reps=20, eval_reps=[], eval_L=[10], test_ids=[2],
    # model_kwargs={"seed" : 160033707},
    # system_kwargs={"seed" : 2144784513},
)
# dd.testing.test_system("dynadojo.systems.lds", L= [20, 100], n_starts=[100,100], t=10, seed = 10, reps=2, eval_reps=[0], eval_L=[20], test_ids=[2])

# Should not run anything because of eval_L out of range
# dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, eval_reps=[0], eval_L=[5] , test_ids=[2])

# # Accept command line arguments
# parser = argparse.ArgumentParser(description='Run experiments')
# parser.add_argument('--system', type=str, default="dynadojo.systems.lds", help='system module')
# parser.add_argument('--n', type=int, nargs='+', default=[1, 10], help='n')