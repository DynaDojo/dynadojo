import unittest
from dynadojo.challenges import FixedComplexity
from dynadojo.systems.lds import LDSystem
from dynadojo.baselines.lr import LinearRegression
from dynadojo.baselines.dnn import DNN
import numpy as np
import pandas as pd
import pandas.testing as pd_testing


systems = [LDSystem] #To test multiple systems, add them to this list
models = [LinearRegression, DNN] #To test multiple models, add them to this list

class TestReproducibility(unittest.TestCase):
    def test_make_init_cond(self):
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                s2 = system(seed=100)
                i1 = s1.make_init_conds_wrapper(n=5)
                i2 = s2.make_init_conds_wrapper(n=5)
                np.testing.assert_array_equal(i1, i2)

    def test_make_data(self):
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                i1 = s1.make_init_conds_wrapper(n=5)
                d1 = s1.make_data_wrapper(i1, timesteps=3, noisy=True)
                
                s2 = system(seed=100)
                i2 = s2.make_init_conds_wrapper(n=5)
                d2 = s2.make_data_wrapper(i2, timesteps=3, noisy=True)
                np.testing.assert_array_equal(d1, d2)

    def test_make_initial_conditions_less_more(self):
        """
        Test that if we make data with more initial conditions, 
        the data starts the same as when we made data with less initial conditions
        """
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                i1 = s1.make_init_conds_wrapper(n=5)

                s2 = system(seed=100)
                i2 = s2.make_init_conds_wrapper(n=10)
                np.testing.assert_array_equal(i1, i2[:5])

    # TODO: Reproducibility test w/ control
                
    # TODO: Improve reproducibility of data generation
    # def test_make_data_less_more(self):
    #     """
    #     Test that if we make data with more initial conditions, 
    #     the data starts the same as when we made data with less initial conditions
    #     """
    #     for system in systems:
    #         with self.subTest(system):
    #             s1 = system(seed=100)
    #             i1 = s1.make_init_conds_wrapper(n=5)
    #             d1 = s1.make_data_wrapper(i1, timesteps=3, noisy=True)

    #             s2 = system(seed=100)
    #             i2 = s2.make_init_conds_wrapper(n=10)
    #             d2 = s2.make_data_wrapper(i2, timesteps=3, noisy=True)
    #             np.testing.assert_array_equal(d1, d2[:5])

class TestReproducibilityModel(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
    
    def test_with_fc(self):
        for model in models:
            with self.subTest(model):
                challenge = FixedComplexity(N=[2], l=2, t=2, 
                                system_cls=LDSystem, reps=1,
                                test_examples=2, test_timesteps=2, verbose=False)
                df1 = challenge.evaluate(model, seed=100, noisy=True,
                        reps_filter=None, 
                        L_filter = None,
                        model_kwargs=None)
                df2 = challenge.evaluate(model, seed=100, noisy=True,
                        reps_filter=None, 
                        L_filter = None,
                        model_kwargs=None)
                cols = ['rep','latent_dim','embed_dim','timesteps','n','error','ood_error','total_cost','system_seed','model_seed']
                df1 = df1[cols]
                df2 = df2[cols]
                self.assertEqual(df1, df2)

    def test_with_reps(self):
        """
        Test that running a single rep gives the same results as running multiple reps,
        when using filters to select a single rep
        """
        for model in models:
            with self.subTest(model):
                challenge = FixedComplexity(N=[2], l=2, t=2, 
                                system_cls=LDSystem, reps=2,
                                test_examples=2, test_timesteps=2, verbose=False)
                df1 = challenge.evaluate(model, seed=100, noisy=True,
                        reps_filter=[1], 
                        L_filter = None,
                        model_kwargs=None)
                df2 = challenge.evaluate(model, seed=100, noisy=True,
                        reps_filter=None, 
                        L_filter = None,
                        model_kwargs=None)
                cols = ['rep','latent_dim','embed_dim','timesteps','n','error','ood_error','total_cost','system_seed','model_seed']
                df1 = df1[cols]
                df2 = df2[cols].loc[df2['rep'] == 1]
                self.assertEqual(df1, df2)


if __name__ == '__main__':
    unittest.main()