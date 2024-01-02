import numpy as np
import pandas as pd
from statistical_functions import portfolio_covar
from general_functions import copy_dict

def get_weights_per_contract(
    notional_exposure_per_contract : dict,
    capital : float) -> dict:
    
    instruments = list(notional_exposure_per_contract.keys())

    weights_per_contract = {}

    for instrument in instruments:
        weights_per_contract[instrument] = notional_exposure_per_contract[instrument] / capital

    return weights_per_contract

def convert_positions_to_weights(
    positions : dict, 
    weights_per_contract : dict) -> dict:

    instruments = list(positions.keys())

    position_weights = {}

    for instrument in instruments:
        position_weights[instrument] = positions[instrument] * weights_per_contract[instrument]

    return position_weights

def zero_weights(
    instruments : list) -> dict:

    weights = {}

    for instrument in instruments:
        weights[instrument] = 0.0
    
    return weights

def convert_costs_to_weights(
    costs_per_contract : dict,
    weights_per_contract : dict,
    capital : float) -> dict:

    instruments = list(costs_per_contract.keys())

    costs_per_contract_weights = {}

    for instrument in instruments:
        costs_per_contract_weights[instrument] = costs_per_contract[instrument] / capital / weights_per_contract[instrument]

    return costs_per_contract_weights

def get_cost_penalty(
    proposed_positions_weights : dict,
    held_positions_weights : dict,
    costs_per_contract_weights : dict,
    cost_penalty_scale = 10) -> float:

    instruments = list(proposed_positions_weights.keys())

    cost_of_all_trades = 0.0

    for instrument in instruments:
        cost_of_all_trades += abs(held_positions_weights[instrument] - proposed_positions_weights[instrument]) * costs_per_contract_weights[instrument]

    return cost_penalty_scale * cost_of_all_trades

def get_tracking_error(
    covariance_matrix : np.array,
    ideal_position_weights : dict,
    proposed_solution : dict,
    cost_penalty : float,
    instruments : list) -> float:

    tracking_error_weights = []

    for instrument in instruments:
        tracking_error_weights.append(proposed_solution[instrument] - ideal_position_weights[instrument])

    tracking_error_weights = np.array(tracking_error_weights)

    tracking_error_radicand = tracking_error_weights.dot(covariance_matrix).dot(tracking_error_weights)
    tracking_error = np.sqrt(tracking_error_radicand)

    tracking_error += cost_penalty

    return tracking_error

def get_optimized_weights(
    held_position_weights : dict,
    ideal_position_weights : dict,
    weights_per_contract : dict,
    costs_per_contract_weights : dict,
    covariance_matrix : np.array,
    instruments : list) -> dict:

    # First proposed solution is zero contracts for each instrument
    proposed_solution = zero_weights(instruments)

    cost_penalty = get_cost_penalty(proposed_solution, held_position_weights, costs_per_contract_weights)

    tracking_error = get_tracking_error(covariance_matrix, ideal_position_weights, proposed_solution, cost_penalty, instruments)
    
    while True:
        previous_solution = copy_dict(proposed_solution)
        best_tracking_error = tracking_error

        best_instrument = None

        for instrument in instruments:
            proposed_solution = copy_dict(previous_solution)

            # if the ideal position weight is positive, we want to increase our current weight
            if (ideal_position_weights[instrument] > 0):
                proposed_solution[instrument] = proposed_solution[instrument] + weights_per_contract[instrument]
            # else: decrease the current weight
            else:
                proposed_solution[instrument] = proposed_solution[instrument] - weights_per_contract[instrument]

            cost_penalty = get_cost_penalty(proposed_solution, held_position_weights, costs_per_contract_weights)
            tracking_error = get_tracking_error(covariance_matrix, ideal_position_weights, proposed_solution, cost_penalty, instruments)

            if tracking_error < best_tracking_error:
                best_tracking_error = tracking_error
                best_instrument = instrument

        proposed_solution = copy_dict(previous_solution)

        # if there is no best instrument, move on
        if best_instrument is None:
            break

        # if the ideal position weight is positive, we want to increase our current weight
        if (ideal_position_weights[best_instrument] > 0):
            proposed_solution[best_instrument] = proposed_solution[best_instrument] + weights_per_contract[best_instrument]
            continue

        # else: decrease the current weight
        proposed_solution[best_instrument] = proposed_solution[best_instrument] - weights_per_contract[best_instrument]

    return proposed_solution

def get_buffered_trades(
    held_positions : dict,
    optimized_positions : dict,
    held_portfolio_tracking_error : float,
    risk_target : float,
    instruments : list,
    asymmetric_buffer : float = 0.05) -> dict:
    
    required_trades = {}

    tracking_error_buffer = risk_target * asymmetric_buffer

    # if the current portfolio is good enough, do no trades
    if (held_portfolio_tracking_error < tracking_error_buffer):
        for instrument in instruments:
            required_trades[instrument] = 0.0

        return required_trades

    adjustment_factor = max((held_portfolio_tracking_error - tracking_error_buffer) / held_portfolio_tracking_error, 0.0)

    for instrument in instruments:
        required_trades[instrument] = round(adjustment_factor * (optimized_positions[instrument] - held_positions[instrument]))
    
    return required_trades

def get_optimized_positions(
    held_positions : dict,
    ideal_positions : dict,
    notional_exposures_per_contract : dict,
    capital : float,
    costs_per_contract : dict,
    returns_df : pd.DataFrame,
    risk_target : float) -> dict:
    
    instruments = list(ideal_positions.keys())
    instruments.sort()

    weights_per_contract = get_weights_per_contract(notional_exposures_per_contract, capital)
    
    ideal_position_weights =  convert_positions_to_weights(ideal_positions, weights_per_contract)

    costs_per_contract_weights = convert_costs_to_weights(costs_per_contract, weights_per_contract, capital)

    held_position_weights = convert_positions_to_weights(held_positions, weights_per_contract)

    covariance_matrix = portfolio_covar(returns_df, instruments)

    optimized_weights = get_optimized_weights(held_position_weights, ideal_position_weights, weights_per_contract, costs_per_contract_weights, covariance_matrix, instruments)

    # get tracking error of optimized weights on held positions 
    # NOTE a 0 cost penalty is used since cost has already been figured into the positions
    held_portfolio_tracking_error = get_tracking_error(covariance_matrix, optimized_weights, held_positions, 0.0, instruments)

    optimized_positions = {}

    for instrument in instruments:
        optimized_positions[instrument] = optimized_weights[instrument] / weights_per_contract[instrument]

    buffered_trades = get_buffered_trades(held_positions, optimized_positions, held_portfolio_tracking_error, risk_target, instruments)

    buffered_positions = {}

    # get the positions we need to have after the buffered trades
    for instrument in instruments:
        buffered_positions[instrument] = held_positions[instrument] + buffered_trades[instrument]

    return buffered_positions
