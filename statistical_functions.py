import numpy as np
import pandas as pd
from math import sqrt
from constants import BUSINESS_DAYS_IN_YEAR, BUSINESS_DAYS_IN_TEN_YEARS
from enum import Enum


class Periods(Enum):
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 20
    QUARTERLY = 60
    YEARLY = 256

def SMA( 
        lst : list[float], 
        span : int = -1
    ) -> float:

    if (span == -1):
        return np.nanmean(lst)
    
    return np.nanmean(lst[-span])


def SR(
        returns : list[float]) -> float:
    mean = np.nanmean(returns)

    return mean / std(returns)


def EWMA ( 
        lst : list[float], 
        span : int = None,
        alpha : float = None,
        threshold : int = 100) -> float:
    """Returns Exponentially Weighted Moving Average given span"""

    #Carver's fromula
    #@ EWMA(t) = α(1 - α)⁰ * y(t) + α(1 - α)¹ * y(t-1) + α(1-α)² * y(t-2) + ... + α(1-α)ⁿ * y(t-n)
    # where alpha is 2 / (span + 1) & n is the length of the list

    # I prefer to use the standard EWMA instead of Carver's 
    # as a it results in the mean being the same as the SMA when all the values are equal
    #@ EWMA(t) = α * y(t) + (1 - α) * EWMA(t-1)

    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")
    

    if alpha is None:
        alpha : float = 2 / (span + 1)


    lst_len : int = len(lst)
    last_IDX : int = lst_len - 1

    if (lst_len <= threshold):
        return SMA(lst)
    
    ewma : float = SMA(lst[:threshold])


    for n in range(threshold, lst_len):
        ewma = alpha * lst[n] + (1 - alpha) * ewma
        # ewma += (alpha * (1-alpha)**n * lst[last_IDX - n])

    return ewma


def std(
        lst : list[float],
        span : int = -1,
        annualize : bool = False) -> float:
    
    """Returns the standard deviation of a list of values"""

    # get the lesser of the two values
    span = min(len(lst), span)

    reduced_lst : list[float] = lst[:-span] if (span != -1) else lst

    xbar : float = np.mean(reduced_lst)

    numerator : float = 0

    for x in reduced_lst:
        numerator += (x - xbar)**2

    lst_len : int = len(reduced_lst)

    if lst_len == 1:
        return 0

    standard_deviation : float = sqrt(numerator / (len(reduced_lst) - 1))

    factor = 1

    if (annualize is True):
        factor = sqrt(BUSINESS_DAYS_IN_YEAR)

    return standard_deviation * factor 


def VAR(
    lst : list[float],
    span : int = -1,
    annualize : bool = False) -> float:

    return std(lst=lst, span=span, annualize=annualize)**2


def exponentially_weighted_stddev(
        lst : list,
        span : int = None, 
        alpha : float = None, 
        annualize : bool = False,
        threshold : int = 100) -> float:
    """
    """

    #@ given an expoentially weighted moving average, r*
    #@                     _________________________________________________________________________________
    #@ exponential σ(t) = √ α(1 - α)⁰(r(t) - r*)² + α(1 - α)¹(r(t-1) - r*)² + α(1 - α)²(r(t-2) - r*)² + ... 


    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")

    ewma : float = EWMA(lst, span=span, alpha=alpha, threshold=threshold)
    
    if alpha is None:
        alpha : float = 2 / (span + 1)

    radicand : float  = 0
    lst_len : int = len(lst)
    last_IDX : int = lst_len - 1

    if (lst_len <= threshold):
        return std(lst=lst, span=span, annualize=annualize)

    # starting value is just the simple variance of the first 100 values (threshold). Variance is the radicand of the stddev formula
    radicand = VAR(lst[:threshold])

    for n in range(threshold, lst_len):
        radicand += (alpha * (1 - alpha)**n * (lst[last_IDX - n] - ewma)**2) 

    ew_stddev : float = sqrt(radicand)

    factor = 1
    if (annualize is True):
        factor = sqrt(BUSINESS_DAYS_IN_YEAR)

    return ew_stddev * factor


def correlation(
    returns_X : pd.DataFrame,
    returns_Y : pd.DataFrame) -> float:
    """Calculates a correlation (rho) between two DataFrames where each dataframe had a "Date" column"""

    rho = 0.0

    # Try to merge the two dataframes on the date column
    try:
        merged_df = pd.merge(returns_X, returns_Y, on="Date", how="inner")
        rho = merged_df.iloc[:,1].corr(merged_df.iloc[:,2])
        
    # If not just merge them on the index
    except KeyError:
        merged_df = pd.merge(returns_X, returns_Y, left_index=True, right_index=True, how="inner")
        rho = merged_df.iloc[:,0].corr(merged_df.iloc[:,1])

    return rho


def correlation_matrix(
    returns_df : pd.DataFrame,
    period : Periods,
    window : int,
    tickers : list = None) -> np.array:

    periodic_returns_df = pd.DataFrame()
    
    if tickers is None:
        tickers = returns_df.columns.tolist()
        tickers.sort()

    for ticker in tickers:
        returns = returns_df[ticker].tolist()

        # groups them and takes the recent window backwards
        periodic_returns_df[ticker] = [sum(returns[x : x + period.value])
                                for x in range(0, len(returns), period.value)][:-window]

    return periodic_returns_df.corr()


def rolling_std(
    returns : pd.DataFrame,
    ten_year_weight : float = 0.3) -> float:
    """Calculates a rolling standard deviation for a given dataframe with weighting on the annualized stddev and 10 year average"""

    annualized_stddevs = []
    ten_year_averages = []

    # max values included in ew_stddev, this should expedite the process
    maximum_values = 100
    
    weighted_stddevs = []

    for n, val in enumerate(returns.tolist()):
        start = max(0, n - maximum_values)
        annualized_stddev = exponentially_weighted_stddev(returns[start:n+1], span=32, annualize=True)

        annualized_stddevs.append(annualized_stddev)

        if n < BUSINESS_DAYS_IN_TEN_YEARS:
            ten_year_average = np.mean(annualized_stddevs[:n+1])
            
        else:
            ten_year_average =np.mean(annualized_stddevs[n-BUSINESS_DAYS_IN_TEN_YEARS:n+1])

        ten_year_averages.append(ten_year_average)

        weighted_stddev = ten_year_weight * ten_year_average + (1 - ten_year_weight) * annualized_stddev

        weighted_stddevs.append(weighted_stddev)

    return weighted_stddevs


def portfolio_covar( 
    position_percent_returns : pd.DataFrame,
    tickers = None) -> np.array:
    """Calculates a covariance matrix as outlined by Carver on pages 606-607"""

    #@ Σ = σ.ρ.σᵀ = σσᵀ ⊙ ρ (using Hadamard product) = Diag(σ) * ρ * Diag(σ)
    #@ where:
    #@ ρ is the correlation matrix
    #@ σ is the vector of annualized estimates of % standard deviations 
    #@ use 32 day span for standard deviations
    #@ window for equally weighted correlation matrix of 52 weeks


    stddev_lst = []

    if tickers is None:
        tickers = position_percent_returns.columns.tolist()
        tickers.sort()

    for ticker in tickers:
        # get the most recent value
        rolling_stddev = rolling_std(position_percent_returns[ticker])[-1]
        stddev_lst.append(rolling_stddev)

    # this is in the same order as the corr_matrix
    stddev_array = np.array(stddev_lst)

    corr_matrix = correlation_matrix(position_percent_returns, Periods.WEEKLY, 52, tickers)

    covar = np.dot(np.dot(np.diag(stddev_array), corr_matrix), np.diag(stddev_array))

    return covar


def portfolio_stddev(
        position_weights : pd.DataFrame,
        position_percent_returns : pd.DataFrame) -> float:
    
    #@                _______
    #@ Portfolio σ = √ w Σ wᵀ
    #@ w is the vector of positions weights, and Σ is the covariance matrix of percent returns 

    tickers : list = position_weights.columns.tolist()

    weights_lst : list = []

    # gets the weights for each instrument
    for ticker in tickers:
        weights_lst.append(position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

    weights = np.array(weights_lst)

    weights_T = weights.transpose()

    covariance_matrix = portfolio_covar(position_percent_returns)

    radicand : float = np.dot(np.dot(weights, covariance_matrix), weights_T)

    return sqrt(radicand)
