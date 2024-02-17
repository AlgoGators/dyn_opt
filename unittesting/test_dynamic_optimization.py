import sys

# Add parent directory to path
sys.path.append('../DYN_OPT')

import unittest
import pandas as pd
import os
import numpy as np
from datetime import datetime

from general_functions import get_daily_returns
from dyn_opt import DynamicOptimization

class TestDynamicOptimization(unittest.TestCase):
    def test_dyn_opt(self):
        contract_names = ['ES', 'NQ', 'ZN']

        returns_df = pd.read_csv('unittesting/data/percent_returns.csv', index_col=0)

        dates = [datetime.strptime(date, '%d-%m-%y') for date in returns_df.index.tolist()]

        capital = 50_000

        #! Loads the data
        correlation_matrices = {}
        for n, date in enumerate(dates):
            correlation_matrices[date] = np.loadtxt(f"unittesting/data/matrices/matrix{n}.csv", delimiter=",")

        specific_date = dates[-10]

        held_positions = pd.DataFrame.from_dict(
            {specific_date: [0, 0, 0]}, 
            orient='index', 
            columns=['ES', 'NQ', 'ZN'])
        
        ideal_positions = pd.read_csv('unittesting/data/ideal_positions.csv', index_col=0)
        ideal_positions.index = pd.to_datetime(ideal_positions.index, format='%d-%m-%y')

        weight_per_contract = pd.read_csv('unittesting/data/notional_exposure.csv', index_col=0) / capital
        weight_per_contract.index = pd.to_datetime(weight_per_contract.index, format='%d-%m-%y')

        cost_per_contract = pd.read_csv('unittesting/data/costs.csv', index_col=0)
        cost_per_contract.index = pd.to_datetime(cost_per_contract.index, format='%d-%m-%y')

        weighted_cost_per_contract = weight_per_contract * cost_per_contract

        dyn_opt = DynamicOptimization(
            historical_ideal_positions=ideal_positions,
            # historical_percent_returns=returns_df,
            historical_rolling_standard_deviations=returns_df.rolling(window=52).std(),
            historical_notional_exposures=weight_per_contract * capital,
            historical_costs_per_contract=cost_per_contract,
            correlation_matrices=correlation_matrices,
            capital=capital)
        
        # dyn_opt.historical_dynamic_optimized_positions.to_csv('testing/optimized_positions.csv')



if __name__ == '__main__':
    unittest.main(failfast=True)