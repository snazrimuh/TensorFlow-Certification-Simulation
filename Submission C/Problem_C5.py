# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model 
# should be trained to predict the next 24 observations of the 7 variables.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
#
# Desired MAE < 0.1 on the normalized dataset.
# ============================================================================================

import numpy as np
import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()



def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_past + n_future))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda window: (window[:-n_past], window[-n_past:, :1]))


    # YOUR CODE HERE
    return dataset.batch(batch_size).prefetch(1)  # YOUR CODE HERE


# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def solution_C5():
    # Downloads and extracts the dataset to the directory that contains this file.
    download_and_extract_data()
    # Reads the dataset from the csv.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    N_FEATURES = len(df.columns)  # YOUR CODE HERE

    # Normalizes the data
    # DO NOT CHANGE THIS
    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    x_train = data[:split_time]
    x_valid = data[split_time:]

    # DO NOT CHANGE THIS
    BATCH_SIZE = 32
    N_PAST = 24
    N_FUTURE = 24
    SHIFT = 1

    # Code to create windowed train and validation datasets.
    # Complete the code in windowed_dataset.
    train_set = windowed_dataset(
        series=x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT
    )  # YOUR CODE HERE
    valid_set = windowed_dataset(
        series=x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(N_FEATURES, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(N_FEATURES, return_sequences=True)),
        tf.keras.layers.Dense(N_FUTURE)
    ])
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)

    history = model.fit(
        train_set,
        epochs=100,
        validation_data=valid_set,
        batch_size=BATCH_SIZE
    )
    # YOUR CODE HERE
    mae_values = history.history['val_loss']

    mae_values = history.history['mae']  # gunakan 'mae' untuk MAE pada data pelatihan
    val_mae_values = history.history['val_mae']  # gunakan 'val_mae' untuk MAE pada data validasi

    for epoch, mae, val_mae in zip(range(1, len(mae_values) + 1), mae_values, val_mae_values):
        print(f'Epoch {epoch}, Mean Absolute Error (MAE): {mae:.4f}, Val MAE: {val_mae:.4f}')

    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C5()
    model.save("model_C5.h5")