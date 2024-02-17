import pandas as pd
import numpy as np
from datetime import datetime
from statistical_functions import portfolio_covar
import logging

class Parameters:
    cost_penalty_scalar : float = 10
    asymmetric_risk_buffer : float = 0.05
    risk_target : float = 0.20
    starting_date = 500

class ConformData():
    def convert_to_datetime(self, df : pd.DataFrame, format='%d-%m-%y'):
        df.index = pd.to_datetime(df.index, format=format)
        return df

    def dropna(self, df : pd.DataFrame) -> pd.DataFrame:
        return df.dropna()
    
    def conform_data(self, df : pd.DataFrame, format='%d-%m-%y') -> pd.DataFrame:
        df = self.convert_to_datetime(df, format=format)
        df = self.dropna(df)
        return df

    def shared_indices(self, dfs: list) -> list:
        shared_indices = dfs[0].index
        for df in dfs[1:]:
            shared_indices = shared_indices.intersection(df.index)

        return shared_indices
    
    def match_dataframes_to_indices(self, dfs: list) -> list:
        shared_indices = self.shared_indices(dfs)

        return [df.loc[shared_indices] for df in dfs]

class DynamicOptimization():
    def __init__(
            self,
            historical_ideal_positions : pd.DataFrame,
            # historical_percent_returns : pd.DataFrame,
            historical_rolling_standard_deviations : pd.DataFrame,
            historical_notional_exposures : pd.DataFrame,
            historical_costs_per_contract : pd.DataFrame,
            correlation_matrices : dict,
            capital : float) -> None:
        
        # Load the data in and adjust to datetime and dropna
        self.historical_ideal_positions : pd.DataFrame = ConformData().conform_data(historical_ideal_positions)
        # self.historical_percent_returns = ConformData.conform_data(historical_percent_returns)
        self.historical_rolling_standard_deviations : pd.DataFrame = ConformData().conform_data(historical_rolling_standard_deviations)
        self.historical_notional_exposures : pd.DataFrame = ConformData().conform_data(historical_notional_exposures)
        self.historical_costs_per_contract : pd.DataFrame = ConformData().conform_data(historical_costs_per_contract)

        self.historical_ideal_positions, self.historical_rolling_standard_deviations, self.historical_notional_exposures, self.historical_costs_per_contract = ConformData().match_dataframes_to_indices([self.historical_ideal_positions, self.historical_rolling_standard_deviations, self.historical_notional_exposures, self.historical_costs_per_contract])

        self.correlation_matrices = correlation_matrices

        self.capital = capital
        
        # Effectively rows and columns
        self.dates = self.historical_ideal_positions.index.tolist()
        self.contract_names = self.historical_ideal_positions.columns.tolist()

        # Calculated tables
        self.historical_weight_per_contract : pd.DataFrame
        self.historical_weighted_cost_per_contract : pd.DataFrame
        self.historical_weighted_ideal_positions : pd.DataFrame

        self.historical_held_positions : pd.DataFrame

        self.historical_dynamic_optimized_positions : pd.DataFrame

        # Set tables
        self.set_weight_per_contract()
        self.set_weighted_cost_per_contract()
        self.set_weighted_ideal_positions()

        # Optimize the positions
        self.set_historically_optimized_positions()


    def set_weight_per_contract(self) -> None:
        self.historical_weight_per_contract = self.historical_notional_exposures / self.capital
    
    def set_weighted_cost_per_contract(self) -> None:
        self.historical_weighted_cost_per_contract = self.historical_weight_per_contract * self.historical_costs_per_contract

    def set_weighted_ideal_positions(self) -> None:
        self.historical_weighted_ideal_positions = self.historical_ideal_positions * self.historical_weight_per_contract

    def set_historically_optimized_positions(self) -> None:
        # Start with held positions for t=0 @ 0
        self.historical_held_positions = pd.DataFrame(0, index=[self.dates[Parameters.starting_date]], columns=self.contract_names)
        
        # Iterate through the dates, @ starting point
        for n, date in enumerate(self.dates[Parameters.starting_date:]):
            optimized_positions = self.optimize_position(date=date)

            if (n == 0):
                self.historical_dynamic_optimized_positions = optimized_positions
            else:
                self.historical_dynamic_optimized_positions = pd.concat([self.historical_dynamic_optimized_positions, optimized_positions])

            if ((n + Parameters.starting_date) == len(self.dates) - 1):
                break

            historical_positions = optimized_positions

            # Gets the next date in the list
            next_date = [self.dates[Parameters.starting_date + n + 1]]
            historical_positions.index = next_date

            self.historical_held_positions = pd.concat([self.historical_held_positions, historical_positions])

    def get_data(self, date : datetime) -> None:
        tple = (self.historical_ideal_positions.loc[[date]],
                self.historical_held_positions.loc[[date]],
                self.historical_rolling_standard_deviations.loc[[date]],
                self.historical_weight_per_contract.loc[[date]],
                self.historical_weighted_cost_per_contract.loc[[date]],
                self.correlation_matrices[date])
        
        return tple

    def optimize_position(self, date : datetime) -> pd.DataFrame:
        ideal_positions, held_positions, rolling_standard_deviations, weight_per_contract, weighted_cost_per_contract, correlation_matrix = self.get_data(date)

        covariance_matrix = portfolio_covar(rolling_standard_deviation=rolling_standard_deviations, correlation_matrix=correlation_matrix)

        daily_optimization = DailyOptimization(
            contract_names = self.contract_names,
            date = date,
            held_positions = held_positions,
            ideal_positions = ideal_positions,
            weight_per_contract = weight_per_contract,
            weighted_costs_per_contract = weighted_cost_per_contract,
            covariance_matrix = covariance_matrix)
        
        return daily_optimization.get_buffered_positions()


class DailyOptimization():
    def __init__(
            self,
            contract_names : list,
            date : datetime,
            held_positions : pd.DataFrame,
            ideal_positions : pd.DataFrame,
            weight_per_contract : pd.DataFrame,
            weighted_costs_per_contract : pd.DataFrame,
            covariance_matrix : np.array) -> None:
        
        self.contract_names = contract_names
        self.date = date
        self.held_positions = held_positions
        self.ideal_positions = ideal_positions
        self.weight_per_contract = weight_per_contract
        self.weighted_costs_per_contract = weighted_costs_per_contract
        self.covariance_matrix = covariance_matrix

        self.weighted_held_positions = self.held_positions * self.weight_per_contract
        self.weighted_ideal_positions = self.ideal_positions * self.weight_per_contract

        self.optimized_weighted_positions : pd.DataFrame

        #! NEEDS name change
        self.optimized_weights : pd.DataFrame
        self.optimized_positions : pd.DataFrame
        self.buffered_positions : pd.DataFrame

        self.set_optimized_weights()
        self.set_optimized_positions()
        self.set_buffered_positions()

    def zero_weights(self):
        return pd.DataFrame.from_dict(
            {self.date: [0 for _ in self.contract_names]}, 
            orient='index', 
            columns=self.contract_names)
    
    def get_cost_penalty(self, weighted_proposed_positions : pd.DataFrame) -> float:
        trading_cost = 0.0

        for contract in self.contract_names:
            position_delta = abs(weighted_proposed_positions.iloc[0][contract] - self.weighted_held_positions.iloc[0][contract]) / self.weight_per_contract.iloc[0][contract]
            trading_cost += position_delta * self.weighted_costs_per_contract.iloc[0][contract]

        return trading_cost
    
    def get_tracking_error(
            self,
            weighted_proposed_positions : pd.DataFrame,
            weighted_comparison_positions : pd.DataFrame,
            cost_penalty : float) -> float:
        
        tracking_error_weights_vector = weighted_proposed_positions.to_numpy() - weighted_comparison_positions.to_numpy()

        tracking_error = np.sqrt(
            tracking_error_weights_vector.dot(self.covariance_matrix).dot(tracking_error_weights_vector.T))

        return tracking_error[0][0] + cost_penalty
    
    def set_optimized_weights(self) -> None:
        weighted_proposed_positions = self.zero_weights()

        cost_penalty : float = self.get_cost_penalty(weighted_proposed_positions=weighted_proposed_positions)

        tracking_error : float = self.get_tracking_error(weighted_proposed_positions, self.weighted_ideal_positions, cost_penalty=cost_penalty)

        best_tracking_error : float = tracking_error
        iteration : int = 0

        while True:
            iteration += 1
            weighted_previous_positions : pd.DataFrame = weighted_proposed_positions.copy(deep=True)

            best_contract = None

            for contract in self.contract_names:
                weighted_proposed_positions = weighted_previous_positions.copy(deep=True)


                if (self.weighted_ideal_positions.iloc[0][contract] > 0):
                    weighted_proposed_positions.at[self.date, contract] = weighted_proposed_positions.loc[self.date][contract] + self.weight_per_contract.loc[self.date][contract]
                else:
                    weighted_proposed_positions.at[self.date, contract] = weighted_proposed_positions.loc[self.date][contract] - self.weight_per_contract.loc[self.date][contract]

                cost_penalty = self.get_cost_penalty(weighted_proposed_positions=weighted_proposed_positions)
                tracking_error = self.get_tracking_error(weighted_proposed_positions, self.weighted_ideal_positions, cost_penalty=cost_penalty)

                if (tracking_error < best_tracking_error):
                    best_tracking_error = tracking_error
                    best_contract = contract

            weighted_proposed_positions = weighted_previous_positions.copy(deep=True)

            if (best_contract is None):
                break

            if (iteration > 1000):
                logging.critical("Iteration limit reached")
                logging.critical(f"Best tracking error: {best_tracking_error}")
                break

            if (self.weighted_ideal_positions.loc[self.date][best_contract] > 0):
                weighted_proposed_positions.at[self.date, best_contract] = weighted_proposed_positions.loc[self.date][best_contract] + self.weight_per_contract.loc[self.date][best_contract]
                continue

            weighted_proposed_positions.at[self.date, best_contract] = weighted_proposed_positions.loc[self.date][best_contract] - self.weight_per_contract.loc[self.date][best_contract]

        self.optimized_weighted_positions = weighted_proposed_positions
            
    def set_optimized_positions(self) -> None:
        self.optimized_positions = self.optimized_weighted_positions / self.weight_per_contract

    def set_buffered_positions(self) -> None:
        # get tracking error of optimized weights on held positions 
        # NOTE a 0 cost penalty is used since cost has already been figured into the positions
        portfolio_tracking_error = self.get_tracking_error(self.optimized_weighted_positions, self.weighted_ideal_positions, 0)

        tracking_error_buffer = Parameters.risk_target * Parameters.asymmetric_risk_buffer

        # if the current portfolio is good enough, do no trades
        if (portfolio_tracking_error < tracking_error_buffer):
            self.buffered_positions = self.held_positions
            return
        
        # if the current portfolio is not good enough, do trades
        adjustment_factor = max((portfolio_tracking_error - tracking_error_buffer) / portfolio_tracking_error, 0.0)

        required_trades = (self.optimized_positions - self.held_positions) * adjustment_factor

        self.buffered_positions = self.held_positions + required_trades

    def get_optimized_weights(self) -> pd.DataFrame:
        return self.optimized_weighted_positions
    
    def get_optimized_positions(self) -> pd.DataFrame:
        return self.optimized_positions
    
    def get_buffered_positions(self) -> pd.DataFrame:
        return self.buffered_positions
