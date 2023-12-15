import sys

# Add parent directory to path
sys.path.append('../DYN_OPT')

import unittest

from dynamic_optimization import *

class TestDynamicOptimization(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_optimal_position(self):
        expected_result = {'ES' : 1, 'ZN' : 0}
        
        optimal_positions = {'ZF' : 0.4, 'ZN' : 0.9, 'MES': 3.1}
        notional_exposures_per_contract = {'ZF' : 110_000, 'ZN' : 120_000, 'MES': 20_000}
        capital = 500_000
        costs_per_contract = {'ZF' : 5.50, 'ZN' : 11.50, 'MES': 0.875}

        result = get_optimized_positions(
            optimal_positions=optimal_positions, 
            notional_exposures_per_contract=notional_exposures_per_contract,
            capital=capital, 
            costs_per_contract=costs_per_contract)
        
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main(failfast=True)