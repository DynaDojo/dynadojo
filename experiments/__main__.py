import dynadojo as dd
print("hello")
dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, eval_reps=None)
dd.testing.test_system("dynadojo.systems.lds", seed = 10, reps=2, eval_reps=[1], eval_L=[4])