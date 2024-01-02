import sys

# Add parent directory to path
sys.path.append('../DYN_OPT')

import unittest
import pandas as pd

from general_functions import get_daily_returns
import dyn_opt

class TestDynamicOptimization(unittest.TestCase):
    def setUp(self):
        MES_returns = get_daily_returns(pd.read_csv('unittesting/data/_MES_Data.csv'), return_column="MES")
        ZF_returns = get_daily_returns(pd.read_csv('unittesting/data/_ZF_Data.csv'), return_column="ZF")
        ZN_returns = get_daily_returns(pd.read_csv('unittesting/data/_ZN_Data.csv'), return_column="ZN")

        self.returns_df = MES_returns.merge(ZF_returns, on='Date', how="inner").merge(ZN_returns, on='Date', how="inner")

    def test_dyn_opt_neg(self):
        expected_result = {'ZF': 0, 'ZN': 1, 'MES': -3}
        
        ideal_positions = {'ZF' : 0.4, 'ZN' : 0.9, 'MES': -3.1}
        notional_exposures_per_contract = {'ZF' : 110_000, 'ZN' : 120_000, 'MES': 20_000}
        capital = 500_000
        costs_per_contract = {'ZF' : 5.50, 'ZN' : 11.50, 'MES': 0.875}

        currently_held_positions = {'ZF': 0, 'ZN': 0, 'MES': 3}

        risk_target = 0.20

        result = dyn_opt.get_optimized_positions(
            held_positions=currently_held_positions,
            ideal_positions=ideal_positions,
            notional_exposures_per_contract=notional_exposures_per_contract,
            capital=capital,
            costs_per_contract=costs_per_contract,
            returns_df=self.returns_df,
            risk_target=risk_target)
        
        self.assertEqual(result, expected_result)

    def test_dyn_opt_pos(self):
        expected_result = {'ZF': 0, 'ZN': 1, 'MES': 3}
        
        ideal_positions = {'ZF' : 0.4, 'ZN' : 0.9, 'MES': 3.1}
        notional_exposures_per_contract = {'ZF' : 110_000, 'ZN' : 120_000, 'MES': 20_000}
        capital = 500_000
        costs_per_contract = {'ZF' : 5.50, 'ZN' : 11.50, 'MES': 0.875}

        currently_held_positions = {'ZF': 0, 'ZN': 1, 'MES': 2}

        risk_target = 0.20

        result = dyn_opt.get_optimized_positions(
            held_positions=currently_held_positions,
            ideal_positions=ideal_positions,
            notional_exposures_per_contract=notional_exposures_per_contract,
            capital=capital,
            costs_per_contract=costs_per_contract,
            returns_df=self.returns_df,
            risk_target=risk_target)
        
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main(failfast=True)