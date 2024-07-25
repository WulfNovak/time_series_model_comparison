from numpy import ascontiguousarray

def rescale_prediction(rescale_object, predicted_values):
    """
    Rescale a vector of predicted values that have been scaled by the general_scale function.

    Args:
    rescale_object: Object created by general_scale function.
                    Gives key information for rescaling the predicted values.                    
    predicted_values: Vector of modeled predicted values of target variable that was scaled by general_scale function.

    Returns:
    Statement of variables that have been standardized or normalized,
    list of non-numeric vars, and resulting data.
    Sets rescale_object to global environment if target_variable is specified.
    """

    predicted_values = np.ascontiguousarray(predicted_values)

    if rescale_object[2] == 'standardized':

        mean = rescale_object[0]
        std = rescale_object[1]

        rescaled_prediction = np.ascontiguousarray(predicted_values * std + mean)
      
    elif rescale_object[2] == 'normalized':
        
        min = rescale_object[0]
        max = rescale_object[1]
    
        rescaled_prediction = np.ascontiguousarray(predicted_values * (min - max) + min)   
    
    return rescaled_prediction