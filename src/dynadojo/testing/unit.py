import unittest
from dynadojo.systems.lds import LDSystem
import numpy as np

systems = [LDSystem] #To test multiple systems, add them to this list


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

    

if __name__ == '__main__':
    unittest.main()