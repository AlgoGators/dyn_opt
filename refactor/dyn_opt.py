import pandas as pd
import numpy as np

def get_notional_exposure_per_contract(unadj_prices : pd.DataFrame, multipliers : pd.DataFrame) -> pd.DataFrame:
    notional_exposure_per_contract = unadj_prices.apply(lambda col: col * multipliers.loc['Multiplier', col.name])
    return notional_exposure_per_contract.abs()

def get_weight_per_contract(notional_exposure_per_contract : pd.DataFrame, capital : float) -> pd.DataFrame:
    return notional_exposure_per_contract / capital

def get_cost_penalty(x : np.ndarray, y : np.ndarray, weighted_cost_per_contract : np.ndarray, cost_penalty_scalar : int) -> float:
    """Finds the trading cost to go from x to y, given the weighted cost per contract and the cost penalty scalar"""

    #* Should never activate but just in case
    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.float64)
    weighted_cost_per_contract = np.array(weighted_cost_per_contract).astype(np.float64)
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    weighted_cost_per_contract[np.isnan(weighted_cost_per_contract)] = 0


    trading_cost = np.abs(x - y) * weighted_cost_per_contract

    return trading_cost.sum() * cost_penalty_scalar

def get_portfolio_tracking_error_standard_deviation(x : np.ndarray, y : np.ndarray, covariance_matrix : np.ndarray, cost_penalty : float) -> float:
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(covariance_matrix).any():
        return ValueError("Input contains NaN values")
    
    tracking_errors = x - y

    print(tracking_errors)

    radicand = tracking_errors.dot(covariance_matrix).dot(tracking_errors.T)

    #* deal with negative radicand (really, REALLY shouldn't happen); just return 100% TE
    #? maybe its a good weight set but for now, it's probably safer this way
    if radicand < 0:
        return 1.0
    
    return np.sqrt(radicand) + cost_penalty

def covariance_row_to_matrix(row : np.ndarray) -> np.ndarray:
    num_instruments = int(np.sqrt(2 * len(row)))
    matrix = np.zeros((num_instruments, num_instruments))

    idx = 0
    for i in range(num_instruments):
        for j in range(i, num_instruments):
            matrix[i, j] = row[idx]
            matrix[j, i] = row[idx]
            idx += 1

    return matrix

#x = np.ndarray( [AA,AB,BB])
covar = np.array([1 ,2 ,3 ])
covar = covariance_row_to_matrix(covar)

print(covar)


x = np.array([-2,2])
y = np.array([1,3])

TE = get_portfolio_tracking_error_standard_deviation(x, y, covar, 0.0)
print(TE)