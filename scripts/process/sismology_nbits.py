# Parameterized version of sismology_8bits.py supporting n bits per output byte.
import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta

def extract_nbits_from_vars(lat_val, lon_val, depth_val, mag_val, n_bits):
    # n_bits must be a multiple of 4
    bits_per_var = n_bits // 4
    # Convert to int, fallback to 0
    def safe_int(x):
        try:
            return int(float(x))
        except:
            return 0
    lat_int = safe_int(lat_val)
    lon_int = safe_int(lon_val)
    depth_int = safe_int(depth_val)
    mag_int = safe_int(mag_val)
    # Extract bits, MSB to LSB for each var
    lat_bits = [(lat_int >> i) & 1 for i in reversed(range(bits_per_var))]
    lon_bits = [(lon_int >> i) & 1 for i in reversed(range(bits_per_var))]
    depth_bits = [(depth_int >> i) & 1 for i in reversed(range(bits_per_var))]
    mag_bits = [(mag_int >> i) & 1 for i in reversed(range(bits_per_var))]
    # Concatenate
    bits = lat_bits + lon_bits + depth_bits + mag_bits
    return bits

def main():
    parser = argparse.ArgumentParser(description="Extract n bits from sismology data per minute.")
    parser.add_argument("--n-bits", "-n", type=int, default=8, help="Number of bits to extract per minute (must be multiple of 4, default 8)")
    args = parser.parse_args()
    n_bits = args.n_bits
    assert n_bits % 4 == 0, "n_bits must be a multiple of 4!"
    input_file = 'raw_data/sismology/sismology_data.parquet'
    output_file = f'processed_data/sismology/nbits_{n_bits}.bin'
    os.makedirs('processed_data/sismology', exist_ok=True)
    print(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    df = df[df['Date'] >= '2020']
    print("Parsing and sorting timestamps...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    start_time = df['Date'].iloc[0].replace(second=0, microsecond=0)
    end_time = df['Date'].iloc[-1].replace(second=0, microsecond=0)
    total_minutes = int((end_time - start_time).total_seconds() // 60) + 1
    latitudes = df['Latitude'].astype(str).values
    longitudes = df['Longitude'].astype(str).values
    depths = df['Depth'].astype(str).values
    magnitudes = df['Magnitude'].astype(str).values
    timestamps = df['Date'].values
    output_bytes = bytearray()
    event_idx = 0
    for i in range(total_minutes):
        minute_time = start_time + timedelta(minutes=i)
        while (event_idx + 1 < len(timestamps)) and (timestamps[event_idx + 1] <= np.datetime64(minute_time)):
            event_idx += 1
        lat_val = latitudes[event_idx]
        lon_val = longitudes[event_idx]
        depth_val = depths[event_idx]
        mag_val = magnitudes[event_idx]
        bits = extract_nbits_from_vars(lat_val, lon_val, depth_val, mag_val, n_bits)
        # Write each bit as a single byte (0x00 or 0x01)
        output_bytes.extend([bit for bit in bits])
        if i % (total_minutes // 100 or 1) == 0:
            print(f'Progress: {i * 100 // total_minutes}% ({i}/{total_minutes})')
    # Write bits as a true bitstream: pack bits into bytes, in extraction order
    out_bytes = bytearray()
    for i in range(0, len(output_bytes), 8):
        chunk = output_bytes[i:i+8]
        byte = 0
        for j, bit in enumerate(chunk):
            byte |= (bit << j)
        out_bytes.append(byte)
    with open(output_file, 'wb') as f:
        f.write(out_bytes)
    print(f'Wrote {len(out_bytes)} bytes ({len(output_bytes)} bits) to {output_file}')

if __name__ == "__main__":
    main()
