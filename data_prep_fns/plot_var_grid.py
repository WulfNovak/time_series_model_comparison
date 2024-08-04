
import matplotlib.pyplot as plt
import seaborn as sb

# Function to plot KDE plots for given variables
def plot_var_grid(dataframe, variables, colors, subplot_x, subplot_y, plot_type): 
    """
    Plot grid of plots for vector of variables
    
    Args:
        dataframe: pandas dataframe,
        variables: list of variables to plot,
        colors: list of colors for each variable,
        subplot_x: int, number of subplots in x direction,
        subplot_y: int, number of subplots in y direction,
        plot_type: str, type of plot to create (lineplot, otherwise kde)

    Returns: grid of lineplots or kde plots
    """
    sb.set_theme()

    # Create a grid of subplots
    fig, axs = plt.subplots(subplot_x, subplot_y, 
                            figsize = (subplot_x * 3, subplot_y * 5), 
                            constrained_layout = True)
    
    # Get date variable
    date = None
    for column in dataframe.columns:
        if (dataframe.index.inferred_type == "datetime64") == True:
            date = dataframe.index.name
        elif dataframe[column].dtype == 'datetime64[ns]': 
            date = column
            # Will need to handle case where there are multiple date columns
        else: 
            TypeError('No datetime64[ns] column found in dataframe')
    
    # Loop through the variables and plot each one for given plot_type
    
    if plot_type == 'lineplot':
        for i, (ax, var, color) in enumerate(zip(axs.flat, variables, colors)):
            sb.lineplot(dataframe, x = f'{date}', y = var, ax = ax, color = color)
            ax.set_title(f'Time Series of {var}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 20, ha = 'right')
            
    else: 
        for i, (ax, var, color) in enumerate(zip(axs.flat, variables, colors)):
            sb.kdeplot(dataframe, x = var, ax = ax, fill = True, color = color)
            ax.set_title(f'Distribution of {var}')

    #plt.xticks(rotation = 28)
    plt.show(); 