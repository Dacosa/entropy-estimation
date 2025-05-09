import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
INPUT_FILE = os.path.join("raw_data", "sismology", "sismology_data.parquet")
OUTPUT_DIR = os.path.join("processed_data", "sismology")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "8bits.bin")

# Load data
print(f"Loading data from {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)
df = df[df['Date'] >= '2020']  # Only from 2020 onwards

# Parse timestamps
print("Parsing and sorting timestamps...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')

# Prepare for per-minute sampling
start_time = df['Date'].iloc[0].replace(second=0, microsecond=0)
end_time = df['Date'].iloc[-1].replace(second=0, microsecond=0)
total_minutes = int((end_time - start_time).total_seconds() // 60) + 1

# Prepare arrays for fast lookup
latitudes = df['Latitude'].astype(str).values
longitudes = df['Longitude'].astype(str).values
depths = df['Depth'].astype(str).values
magnitudes = df['Magnitude'].astype(str).values
timestamps = df['Date'].values

# Initialize output
output_bytes = bytearray()
event_idx = 0

for i in range(total_minutes):
    minute_time = start_time + timedelta(minutes=i)
    # Move event_idx to the last event <= this minute
    while (event_idx + 1 < len(timestamps)) and (timestamps[event_idx + 1] <= np.datetime64(minute_time)):
        event_idx += 1
    # Get current values (repeat last if no new event)
    lat_val = latitudes[event_idx]
    lon_val = longitudes[event_idx]
    depth_val = depths[event_idx]
    mag_val = magnitudes[event_idx]
    # Convert to numeric, fallback to 0 if conversion fails
    try:
        lat_int = int(float(lat_val))
    except:
        lat_int = 0
    try:
        lon_int = int(float(lon_val))
    except:
        lon_int = 0
    try:
        depth_int = int(float(depth_val))
    except:
        depth_int = 0
    try:
        mag_int = int(float(mag_val))
    except:
        mag_int = 0
    # Take last 2 bits from each value
    lat_bits = lat_int & 0b11
    lon_bits = lon_int & 0b11
    depth_bits = depth_int & 0b11
    mag_bits = mag_int & 0b11
    # Pack into 8 bits: [lat2 | lon2 | depth2 | mag2]
    byte_val = (lat_bits << 6) | (lon_bits << 4) | (depth_bits << 2) | mag_bits
    output_bytes.append(byte_val)
    # Progress print every 1%
    if i % (total_minutes // 100 or 1) == 0:
        print(f'Progress: {i * 100 // total_minutes}% ({i}/{total_minutes})')

with open(OUTPUT_FILE, 'wb') as f:
    f.write(output_bytes)

print(f'Wrote {len(output_bytes)} bytes to {OUTPUT_FILE}')