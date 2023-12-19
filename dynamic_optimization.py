import pandas as pd
import numpy as np
import copy

from statistical_functions import portfolio_covar
from math import sqrt

def get_optimal_portfolio_weights(
    optimal_positions : dict,
    notional_exposures_per_contract : dict,
    capital : float) -> dict:
    """
    Returns a dictionary of weights for each instrument given an optimal position

    Parameters:
    ---
        optimal_positions : dict
            Dictionary of optimal positions for each instrument assuming infinite capital (50MM)
        notional_exposures_per_contract : dict
            Dictionary of the notional exposure per contract for each instrument
        capital : float
            The capital available to be used
    ---
    """

    instruments = list(optimal_positions.keys())

    optimal_portfolio_weights = {}

    weights_per_contract = get_weights_per_contract(notional_exposures_per_contract, capital)

    for instrument in instruments:
        optimal_portfolio_weights[instrument] = optimal_positions[instrument] * weights_per_contract[instrument]

    return optimal_portfolio_weights


def get_weights_per_contract(
    notional_exposures_per_contract : dict,
    capital : float) -> dict:
    """
    Returns a dictionary of weights for a single contract for each instrument compared to the total capital

    Parameters:
    ---
        notional_exposures_per_contract : dict
            Dictionary of the notional exposure per contract for each instrument
            (in same currency as capital)
        capital : float
            The capital available to be used
            (in same currency as notional_exposures_per_contract)
    ---
    """

    instruments = list(notional_exposures_per_contract.keys())

    weights_per_contract = {}

    for instrument in instruments:
        weights_per_contract[instrument] = notional_exposures_per_contract[instrument] / capital
    
    return weights_per_contract


def get_costs_per_contract_in_weight_terms(
    notional_exposures_per_contract : dict,
    capital : float,
    costs_per_contract : dict) -> dict:
    """
    Returns a dictionary of costs per contract for each instrument compared to the total capital

    Parameters:
    ---
        notional_exposures_per_contract : dict
            Dictionary of the notional exposure per contract for each instrument
            (in same currency as capital)
        capital : float
            The capital available to be used
            (in same currency as notional_exposures_per_contract)
        costs_per_contract : dict
            Dictionary of the costs per contract for each instrument (estimate)
    ---
    """

    instruments = list(notional_exposures_per_contract.keys())
    
    costs_per_contract_in_weight_terms = {}

    weights_per_contract = get_weights_per_contract(notional_exposures_per_contract, capital)

    for instrument in instruments:
        costs_per_contract_in_weight_terms[instrument] = costs_per_contract[instrument] / capital / weights_per_contract[instrument]

    return costs_per_contract_in_weight_terms


def convert_positions_to_weight(
    positions : dict,
    weights_per_contract : dict) -> dict:
    """
    Converts a dictionary of positions into a dictionary of weights

    Parameters:
    ---
        positions : dict
            Dictionary of positions for each instrument
        weights_per_contract : dict
            Dictionary of weights per contract for each instrument
    ---
    """

    instruments = list(positions.keys())

    position_weights = {}

    for instrument in instruments:
        position_weights[instrument] = positions[instrument] * weights_per_contract[instrument]

    return position_weights


def zero_weights(
    instruments : list) -> dict:
    """
    Returns a dictionary of zero weights for each instrument (a zero set)
    
    Parameters:
    ---
        instruments : list
            List of instruments
    ---
    """

    weights =  {}

    for instrument in instruments:
        weights[instrument] = 0.0

    return weights


def get_tracking_error(
    covariance_matrix : np.array,
    optimal_portfolio_weights : dict,
    current_weights : dict,
    cost_penalty : float) -> float:
    """
    Returns the tracking error of a given portfolio and the optimal portfolio weights

    Parameters:
    ---
        covariance_matrix : np.array
            The covariance matrix of the portfolio
        optimal_portfolio_weights : dict
            Dictionary of optimal portfolio weights
        current_weights : dict
            Dictionary of current portfolio weights
        cost_penalty : float
            The cost penalty for all trades
    ---
    """

    instruments = list(optimal_portfolio_weights.keys())

    tracking_error_weights = []

    for instrument in instruments:
        tracking_error_weights.append(current_weights[instrument] - optimal_portfolio_weights[instrument])
    
    tracking_error_weights = np.array(tracking_error_weights)

    tracking_error = sqrt(tracking_error_weights.dot(covariance_matrix).dot(tracking_error_weights))
    
    tracking_error += cost_penalty

    return tracking_error


def get_cost_penalty(
    optimized_positions_weights : dict,
    currently_held_positions_weights : dict,
    costs_per_contract_in_weight_terms : dict) -> float:
    """
    Returns the cost penalty given the optimized positions, currently held positions, and costs per contract in weight terms

    Parameters:
    ---
        optimized_positions : dict
            Dictionary of optimized positions for each instrument
        currently_held_positions : dict
            Dictionary of currently held positions for each instrument
        costs_per_contract_in_weight_terms : dict
            Dictionary of costs per contract in weight terms for each instrument
    ---
    """

    instruments = list(optimized_positions_weights.keys())

    cost_of_all_trades = 0.0

    for instrument in instruments:
        cost_of_all_trades += abs(currently_held_positions_weights[instrument] - optimized_positions_weights[instrument]) * costs_per_contract_in_weight_terms[instrument]

    # NOTE Carver uses 50x for the cost penalty but [10-100] is reasonable
    return cost_of_all_trades * 50


def get_optimized_weights(
    instruments : list,
    currently_held_position_weights : dict,
    optimal_portfolio_weights : dict,
    weights_per_contract : dict,
    costs_per_contract_in_weight_terms : dict,
    covariance_matrix : np.array,
    cost_penalty) -> dict:
    """
    Iterates over instruments, with single contract increments to find the best tracking error under a greedy algorithm

    Parameters:
    ---
        instruments : list
            List of instruments
        currently_held_position_weights : dict
            Dictionary of currently held position weights for each instrument
        optimal_portfolio_weights : dict
            Dictionary of optimal portfolio weights for each instrument
        weights_per_contract : dict
            Dictionary of weights per contract for each instrument
        costs_per_contract_in_weight_terms : dict
            Dictionary of costs per contract in weight terms for each instrument
        covariance_matrix : np.array
            The covariance matrix of the portfolio
        cost_penalty : float
            The cost penalty for all trades
    ---
    """
    
    current_weights = zero_weights(instruments)

    tracking_error = get_tracking_error(covariance_matrix, optimal_portfolio_weights, current_weights, cost_penalty)

    while True:
        # have to use a deepcopy because dictionaries are mutable so they'd reference the same MEM address
        previous_weights = copy.deepcopy(current_weights)
        best_tracking_error = tracking_error

        best_instrument = None

        for instrument in instruments:
            # Reset the dictionary to the previous weights
            current_weights = copy.deepcopy(previous_weights)

            current_weights[instrument] = current_weights[instrument] + weights_per_contract[instrument]
            cost_penalty = get_cost_penalty(current_weights, currently_held_position_weights, costs_per_contract_in_weight_terms)
            tracking_error = get_tracking_error(covariance_matrix, optimal_portfolio_weights, current_weights, cost_penalty)
            
            if tracking_error < best_tracking_error:
                best_tracking_error = tracking_error
                best_instrument = instrument


        # set current_weights back to previous_weights and increment the weight for the best instrument
        current_weights = previous_weights

        # check if there was a best instrument
        if best_instrument is not None:
            current_weights[best_instrument] = current_weights[best_instrument] + weights_per_contract[best_instrument]
        else:
            break

    return current_weights


def get_buffered_trades(
    currently_held_positions : dict,
    optimized_positions : dict,
    current_portfolio_tracking_error : float,
    risk_target : float,
    asymmetric_buffer : float = 0.05) -> dict:
    """
    Returns a dictionary of the trades needed to get the current portfolio to upper bound of the buffered optimized position
        e.g. have 14 contracts of MES, buffered optimized position is [10, 12], the trade would be -2
    
    Parameters:
    ---
        currently_held_positions : dict
            Dictionary of currently held positions for each instrument
        optimized_positions : dict
            Dictionary of optimized positions for each instrument
        current_portfolio_tracking_error : float
            The tracking error of the current portfolio against the dynamically optimized portfolio
        risk_target : float
            The risk target for the portfolio, Carver uses 0.20
        asymmetric_buffer : float
            The buffer for the upper bound of the dynamically optimized portfolio 
            (Normally 0.10 is used but that includes an upper and lower bound for the positions whereas we are only buffering the upper bound of tracking error)
    ---
    """
    
    instruments = list(currently_held_positions.keys())
    
    required_trades = {}

    buffer = risk_target * asymmetric_buffer

    # if the current portfolio is good enough, do no trades
    if (current_portfolio_tracking_error < buffer):
        for instrument in instruments:
            required_trades[instrument] = 0.0

        return required_trades
    
    adjustment_factor = max((current_portfolio_tracking_error - buffer) / current_portfolio_tracking_error, 0.0)

    for instrument in instruments:
        required_trades[instrument] = round(adjustment_factor * (optimized_positions[instrument] - currently_held_positions[instrument]))
    
    return required_trades


def get_optimized_positions(
    currently_held_positions : dict,
    optimal_positions : dict,
    notional_exposures_per_contract : dict,
    capital : float,
    costs_per_contract : dict,
    returns_df : pd.DataFrame,
    risk_target : float) -> dict:
    """
    Returns a dictionary of optimized positions given certain capital

    NOTE:
        All currency values must be in the same currency, i.e. convert all exposures/costs to $

    Parameters:
    ---
        currently_held_positions : dict
            Dictionary of currently held positions for each instrument
        optimal_positions : dict
            Dictionary of optimal positions for each instrument assuming infinite capital (50MM)
        notional_exposures_per_contract : dict
            Dictionary of the notional exposure per contract for each instrument
        capital : float
            The capital available to be used
        costs_per_contract : dict
            Dictionary of the costs per contract for each instrument (estimate)
        returns_df : pd.DataFrame
            Historical returns for each instruments (daily)
        risk_target : float
            The risk target for the portfolio, Carver uses 0.20
    ---
    """

    instruments = list(optimal_positions.keys())

    optimal_portfolio_weights = get_optimal_portfolio_weights(optimal_positions, notional_exposures_per_contract, capital)

    costs_per_contract_in_weight_terms = get_costs_per_contract_in_weight_terms(notional_exposures_per_contract, capital, costs_per_contract)

    weights_per_contract = get_weights_per_contract(notional_exposures_per_contract, capital)

    covariance_matrix = portfolio_covar(returns_df)

    currently_held_position_weights = convert_positions_to_weight(currently_held_positions, weights_per_contract)

    cost_penalty = get_cost_penalty(optimal_portfolio_weights, currently_held_position_weights, costs_per_contract_in_weight_terms)

    optimized_weights = get_optimized_weights(instruments, currently_held_position_weights, optimal_portfolio_weights, weights_per_contract, costs_per_contract_in_weight_terms, covariance_matrix, cost_penalty)

    optimized_positions = {}

    for instrument in instruments:
        optimized_positions[instrument] = optimized_weights[instrument] / weights_per_contract[instrument]

    # tracking error of the current portfolio against the dynamically optimized portfolio
    cost_penalty = get_cost_penalty(optimized_weights, currently_held_position_weights, costs_per_contract_in_weight_terms)

    current_portfolio_tracking_error = get_tracking_error(covariance_matrix, optimized_weights, currently_held_position_weights, cost_penalty)

    # the number of trades we need to make
    buffered_trades = get_buffered_trades(currently_held_positions, optimized_positions, current_portfolio_tracking_error, risk_target=risk_target, asymmetric_buffer=0.05)

    buffered_positions = {}

    # get the positions we need to have after the buffered trades
    for instrument in instruments:
        buffered_positions[instrument] = currently_held_positions[instrument] + buffered_trades[instrument]

    return buffered_positions
