from dynascale.challenges.lds import LDSChallenge
from dynascale.baselines.simple import Simple
from dynascale.tasks import FixedTrainSize

def main():
    task = FixedTrainSize(n=3000, L=[2, 3], E=[4], T=[50], C=[0], control_horizons=1, test_examples=10, reps=1,
                          test_timesteps=50, challenge_cls=LDSChallenge)
    data = task.evaluate(model_cls=Simple, fit_kwargs={"epochs": 5})
    task.plot(data)


if __name__ == '__main__':
    main()
