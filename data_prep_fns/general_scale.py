import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def general_scale(data, alpha = 0.1):
    """
    Standardize or Normalize the data depending on if Shapiro-Wilk test is significant

    Args:
    data: Data to be standardized or normalized
    alpha: Significance level for Shapiro-Wilk test (default: alpha = 0.1)

    Returns:
    Statement of variables that have been standardized or normalized,
    and resulting data
    """
    standard = StandardScaler()
    minmax = MinMaxScaler()

    p_values = []
    standardize = []
    normalize = []

    for variable in data.columns:
        _, p_value = shapiro(data[variable])
        p_values.append(p_value)
        if p_value < alpha:
            standardize.append(variable)
        else: 
            normalize.append(variable) 
    
    if len(standardize) == 0:
        data[normalize] = minmax.fit_transform(data[normalize])

    else:
        data[standardize] = standard.fit_transform(data[standardize])
        data[normalize] = minmax.fit_transform(data[normalize])

    print("Standardized Variables: ", standardize)
    print("Min-Max Normalized Variables: ", normalize)

    return data 
    