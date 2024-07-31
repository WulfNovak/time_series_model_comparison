import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def general_scale(data, alpha = 0.05):
    """
    Standardize or Normalize the data depending on if Shapiro-Wilk test is significant.
    Returns object to rescale predicted values if applicable.

    Args:
    data: Data to be standardized or normalized
    target_variable: Variable to be rescaled if applicable (default: np.nan)
    alpha: Significance level for Shapiro-Wilk test (default: alpha = 0.1)

    Returns:
    Statement of variables that have been standardized or normalized,
    list of non-numeric vars, and resulting data.
    Sets rescale_key to global environment if target_variable is specified.
    """
        
    # Separate numeric and non-numeric columns
    numeric = data.select_dtypes(include = np.number)
    non_numeric = data.select_dtypes(exclude = np.number)

    standard = StandardScaler()
    minmax = MinMaxScaler()

    standardize = []
    normalize = []

    # Apply Shapiro-Wilk test to check normality
    for variable in numeric.columns:
        _, p_value = shapiro(numeric[variable])
        if p_value < alpha:
            standardize.append(variable)
        else: 
            normalize.append(variable) 
    
    # Standardize or Normalize based on Shapiro-Wilk test results
    if len(standardize) == 0:
        numeric[normalize] = minmax.fit_transform(numeric[normalize])

    elif len(normalize) == 0:
        numeric[standardize] = standard.fit_transform(numeric[standardize])

    else:
        numeric[standardize] = standard.fit_transform(numeric[standardize])
        numeric[normalize] = minmax.fit_transform(numeric[normalize])

    print("Standardized Variables: ", standardize)
    print("Min-Max Normalized Variables: ", normalize)
    print("Non-Numeric Variables: ", list(non_numeric.columns))

    data = pd.concat([numeric, non_numeric], axis = 1)
        
    return data