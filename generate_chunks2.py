import pickle as pkl
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import psutil
import os

FILENAME = "MetroPT2.csv"

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_MB = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"Current memory usage: {memory_MB:.2f} MB")

def generate_chunks(df, chunk_size, chunk_stride, cols):
    from numpy.lib.stride_tricks import sliding_window_view

    gaps = list((g := df.timestamp.diff().gt(pd.Timedelta(minutes=1)))[g].index)
    c = []
    window_start_date = []
    start = 0
    for gap in gaps:
        tdf = df.iloc[start:gap, :]
        if len(tdf) < chunk_size:
            start = gap
            continue
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])
        c.append(sliding_vals)
        start = gap
    tdf = df.iloc[start:, :]
    if len(tdf) >= chunk_size:
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        c.append(sliding_vals)
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])

    c = np.concatenate(c)
    return c, np.concatenate(window_start_date)

final_metro = pd.read_csv(FILENAME)
correct_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                'Oil_temperature', 'Flowmeter', 'Motor_current','COMP', 'DV_eletric',
                'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
orig_cols = ['oem_io.ANCH1', 'oem_io.ANCH2', 'oem_io.ANCH3',
             'oem_io.ANCH4', 'oem_io.ANCH5', 'oem_io.ANCH6', 'oem_io.ANCH7',
             'oem_io.ANCH8', 'oem_io.DI1', 'oem_io.DI2', 'oem_io.DI3', 'oem_io.DI4',
             'oem_io.DI5', 'oem_io.DI6', 'oem_io.DI7', 'oem_io.DI8']
final_metro.rename({orig_cols[i]: correct_cols[i] for i in range(len(correct_cols))}, inplace=True, axis=1)
final_metro["timestamp"] = pd.to_datetime(final_metro["timestamp"])
final_metro = final_metro.sort_values("timestamp")
final_metro.reset_index(drop=True, inplace=True)

analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                  'Oil_temperature', 'Flowmeter', 'Motor_current']

print("Read dataset")

chunks, chunk_dates = generate_chunks(final_metro, 1800, 60, analog_sensors)

print("Calculated chunks")
print_memory_usage()

# Modified scaling approach
scaler = StandardScaler()
print("scaler init")

# Fit the scaler on the entire dataset
all_data = np.concatenate(chunks, axis=0)
scaler.fit(all_data)
print("Fitted scaler on entire dataset")

# Transform each chunk
scaled_chunks_list = [scaler.transform(chunk) for chunk in chunks]
scaled_chunks = np.array(scaled_chunks_list)
print("Finished scaling chunks")

training_chunks = th.tensor(chunks[np.where(chunk_dates[:, 1] < np.datetime64("2022-06-01T00:00:00.000000000"))[0]])
test_chunks = th.tensor(chunks[np.where(chunk_dates[:, 0] >= np.datetime64("2022-06-01T00:00:00.000000000"))[0]])

print("Separated into training and test")

training_chunk_dates = chunk_dates[np.where(chunk_dates[:,1] < np.datetime64("2022-06-01T00:00:00.000000000"))[0]]
test_chunk_dates = chunk_dates[np.where(chunk_dates[:,0] >= np.datetime64("2022-06-01T00:00:00.000000000"))[0]]

with open("data/training_chunk_dates.pkl", "wb") as pklfile:
    pkl.dump(training_chunk_dates, pklfile)

with open("data/test_chunk_dates.pkl", "wb") as pklfile:
    pkl.dump(test_chunk_dates, pklfile)

with open("data/training_chunks.pkl", "wb") as pklfile:
    pkl.dump(training_chunks, pklfile)

with open("data/test_chunks.pkl", "wb") as pklfile:
    pkl.dump(test_chunks, pklfile)

print("Finished saving")
