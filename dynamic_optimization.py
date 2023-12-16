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
        capital : float
            The capital available to be used
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
        capital : float
            The capital available to be used
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
    current_weights : dict) -> float:
    """
    """

    instruments = list(optimal_portfolio_weights.keys())

    tracking_error_weights = []

    for instrument in instruments:
        tracking_error_weights.append(current_weights[instrument] - optimal_portfolio_weights[instrument])
    
    tracking_error_weights = np.array(tracking_error_weights)

    
    tracking_error = sqrt(tracking_error_weights.dot(covariance_matrix).dot(tracking_error_weights))

    return tracking_error


def get_optimized_positions(
    optimal_positions : dict,
    notional_exposures_per_contract : dict,
    capital : float,
    costs_per_contract : dict,
    returns_df : pd.DataFrame) -> dict:
    """
    Returns a dictionary of optimized positions given certain capital

    Parameters:
    ---
        optimal_positions : dict
            Dictionary of optimal positions for each instrument assuming infinite capital (50MM)
        notional_exposures_per_contract : dict
            Dictionary of the notional exposure per contract for each instrument
        capital : float
            The capital available to be used
        costs_per_contract : dict
            Dictionary of the costs per contract for each instrument (estimate)
    ---
    """

    instruments = list(optimal_positions.keys())

    optimal_portfolio_weights = get_optimal_portfolio_weights(optimal_positions, notional_exposures_per_contract, capital)

    costs_per_contract_in_weight_terms = get_costs_per_contract_in_weight_terms(notional_exposures_per_contract, capital, costs_per_contract)

    weights_per_contract = get_weights_per_contract(notional_exposures_per_contract, capital)

    current_weights = zero_weights(instruments)

    covariance_matrix = portfolio_covar(returns_df)

    tracking_error = get_tracking_error(covariance_matrix, optimal_portfolio_weights, current_weights)

    while True:
        # have to use a deepcopy because dictionaries are mutable so they'd reference the same MEM address
        previous_weights = copy.deepcopy(current_weights)
        best_tracking_error = tracking_error

        best_instrument = None

        for instrument in instruments:
            current_weights[instrument] = current_weights[instrument] + weights_per_contract[instrument]
            tracking_error = get_tracking_error(covariance_matrix, optimal_portfolio_weights, current_weights)

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
