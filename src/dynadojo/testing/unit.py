import unittest
from dynadojo.systems.lds import LDSystem

# Test 1
# - Instantiate 2 systems with the same seed
# - Check that they produce the same data

system = [LDSystem]
class TestReproducibility(unittest.TestCase):

    systems = [LDSystem] #To test multiple systems, add them to this list
    def test_make_init_cond(self):
        for system in system:
            with self.subTest(system):
                s1 = system(seed=100)
                s2 = system(seed=100)
                i1 = s1.make_init_conds(n=5)
                i2 = s2.make_init_conds(n=5)
                self.assertEqual(i1, i2)

    def test_make_data(self):
        for system in system:
            with self.subTest(system):
                s1 = system(seed=100)
                s2 = system(seed=100)
                i1 = s1.make_init_conds(n=5)
                i2 = s2.make_init_conds(n=5)
                d1 = s1.make_data(i1, timesteps=3, noisy=True)
                d2 = s2.make_data(i2, timesteps=3, noisy=True)
                self.assertEqual(d1, d2)
    
    def test_make_data_less_more(self):
        """
        Test that if we make data with more initial conditions, 
        the data starts the same as when we made data with less initial conditions
        """
        for system in system:
            with self.subTest(system):
                s1 = system(seed=100)
                s2 = system(seed=100)
                i1 = s1.make_init_conds(n=5)
                i2 = s2.make_init_conds(n=10)
                d1 = s1.make_data(i1, timesteps=3, noisy=True)
                d2 = s2.make_data(i2, timesteps=3, noisy=True)
                self.assertEqual(d1, d2[:int(5), : , :])