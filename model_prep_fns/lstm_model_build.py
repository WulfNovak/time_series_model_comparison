import tensorflow as tf
import keras
import keras_tuner as kt
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


# https://keras.io/guides/keras_tuner/getting_started/
# x_train, y_train

def lstm_model_build( x_train, y_train, model_type, window_size):
    """
    Inputs: 
    model_type: 'base' - single layer LSTM, 'stacked' - two layer LSTM
    x_train: Training data for LSTM model
    y_train: Target data for LSTM model
    ### Should data be windowed within, out outside of this function? 
    window_size:

    
    Output:
    Trained LSTM model
    # Should anything else be returned?

    Other thoughts / future notes:
    Create function for min and max units based on number of features (ex: 4x features = min nodes, 8x features = max nodes)
    # Think of possible warnings or errors that could be raised
        # Neural Net is too large for backpropogation, use batch_size
    # continue to refine actual params within this function. 
    """
    if model_type == 'base':
        lstm = tf.keras.Sequential() 
        lstm.add(tf.keras.layers.LSTM(
            hp.Choice('units', [128, 182]), 
            activation='relu', 
            input_shape=(window_size, x_train_1.shape[1]))) 
        lstm.add(tf.keras.layers.Dropout(.2)) 
        lstm.add(tf.keras.layers.Dense(1))

    elif model_type == 'stacked':
        lstm = tf.keras.Sequential() 
        lstm.add(tf.keras.layers.LSTM(
            hp.Choice('units_1', [128, 182]), 
            activation='relu', 
            input_shape=(window_size, x_train_1.shape[1]),
            return_sequences=True))
        lstm.add(tf.keras.layers.Dropout(.2)) 
        lstm.add(tf.keras.layers.LSTM( 
            hp.Choice('units_2', [128, 182]), 
            activation='relu')) 
        lstm.add(tf.keras.layers.Dropout(.2)) 
        lstm.add(tf.keras.layers.Dense(1))

    lstm.compile(loss='mse', optimizer='adam')
    
    return lstm

tuner = kt.GridSearch( 
    hypermodel=lstm_model_build,
    objective=kt.Objective('loss', direction = 'min'),
    executions_per_trial=5,
    seed=89,
    directory='keras_tuner_dir', 
    project_name='lstm',
    overwrite=True
)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                            patience=3, 
                                            min_delta=1,
                                            restore_best_weights=True,
                                            start_from_epoch=5) 

# Perform hyperparameter search
tuner.search(
    x=ts_train_windows, 
    shuffle=False,
    epochs=50, # Consider raising this value (100)
    callbacks=[callback]
)