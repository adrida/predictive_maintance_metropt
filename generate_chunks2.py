import pickle as pkl
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import psutil
import os
import gc

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
final_metro = None
print("Calculated chunks")
print_memory_usage()

# Modified scaling approach
scaler = StandardScaler()
print("scaler init")

# Incrementally fit the scaler on each chunk
for chunk in chunks:
    scaler.partial_fit(chunk)
    print("Partial fit done")
    print_memory_usage()
print("Fitted scaler on entire dataset incrementally")

# Process each chunk individually and write to disk immediately
with open("data/training_chunks.pkl", "wb") as train_file, \
     open("data/test_chunks.pkl", "wb") as test_file, \
     open("data/training_chunk_dates.pkl", "wb") as train_dates_file, \
     open("data/test_chunk_dates.pkl", "wb") as test_dates_file:

    for i, (chunk, date) in enumerate(zip(chunks, chunk_dates)):
        scaled_chunk = scaler.transform(chunk)
        if date[1] < np.datetime64("2022-06-01T00:00:00.000000000"):
            pkl.dump(scaled_chunk, train_file)
            pkl.dump(date, train_dates_file)
        else:
            pkl.dump(scaled_chunk, test_file)
            pkl.dump(date, test_dates_file)
        print(f"Processed and saved chunk {i+1}/{len(chunks)}")
        print_memory_usage()

        # Explicitly delete the chunk and date to free up memory
        del chunk
        del date

        # Manually run the garbage collector
        gc.collect()

print("Finished saving")