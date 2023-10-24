from .systems import test_system

# test_id = 0 is FixedComplexity
# test_id = 1 is FixedTrainSize
# test_id = 2 is FixedError


# test_system("dynadojo.systems.lds", seed = 10, reps=2, reps_filter=None, test_ids=[0,1,2])
test_system("dynadojo.systems.lds", seed = 2000, reps=20, reps_filter=[], L_filter=[], test_ids=[0],
        plot = True,
        # model_kwargs={"seed" : 160033707},
        # system_kwargs={"seed" : 2144784513},
)
# test_system("dynadojo.systems.lds", L= [20, 100], n_starts=[100,100], t=10, seed = 10, reps=2, reps_filter=[0], L_filter=[20], test_ids=[2])

# Should not run anything because of L_filter out of range
# dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, reps_filter=[0], L_filter=[5] , test_ids=[2])