# This is a parameterized version of bus_8bits.py, now supporting n bits per output byte.
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

from datetime import datetime, timedelta

def parse_posiciones(pos_str):
    fields = pos_str.strip(';').split(';')
    for l in range(11, 25):
        if len(fields) % l == 0:
            record_length = l
            break
    else:
        record_length = 11  # fallback
    records = [fields[i:i+record_length] for i in range(0, len(fields), record_length)]
    return records, record_length

def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except ValueError:
        return datetime.strptime(dt_str, "%d-%m-%Y %H:%M:%S")

def extract_last_decimal_digit(coord_str):
    try:
        if '.' in coord_str:
            return int(coord_str.strip().split('.')[-1][-1])
        else:
            return 0
    except Exception:
        return 0

def process_single_file(json_path, n_bits):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        request_time = parse_datetime(data['fecha_consulta'])
        bus_latest = {}
        for pos_str in data['posiciones']:
            records, _ = parse_posiciones(pos_str)
            if records and len(records[0]) > 4:
                latest_time_str = records[0][0]
                bus_id = records[0][1]
                lat = records[0][2]
                lon = records[0][3]
                try:
                    latest_time = parse_datetime(latest_time_str)
                except Exception:
                    continue
                if (bus_id not in bus_latest) or (latest_time > bus_latest[bus_id][0]):
                    bus_latest[bus_id] = (latest_time, lat, lon)
        # Only keep buses active in last 2 minutes
        active_buses = [
            (bus_id, info[0], info[1], info[2])
            for bus_id, info in bus_latest.items()
            if request_time - info[0] <= timedelta(minutes=2)
        ]
        # Take n//2 buses, each gives 2 bits (lat LSB, lon LSB)
        buses_needed = (n_bits + 1) // 2
        bits = []
        for bus in active_buses[:buses_needed]:
            lat_digit = extract_last_decimal_digit(bus[2])
            lon_digit = extract_last_decimal_digit(bus[3])
            bits.append(lat_digit % 2)
            bits.append(lon_digit % 2)
        # Truncate or pad to exactly n_bits
        bits = bits[:n_bits] + [0] * max(0, n_bits - len(bits))
        # Write each bit as a single byte (0x00 or 0x01)
        out_bytes = bytearray(bits)
        return out_bytes, bits, str(json_path)
    except Exception as e:
        return None, None, f"Error processing {json_path}: {e}"

def process_folder_parallel(input_folder, output_file_path, n_bits):
    input_folder = Path(input_folder)
    all_json_files = sorted(input_folder.rglob('*.json'))
    print(f"Found {len(all_json_files)} JSON files in {input_folder}")
    results = [None] * len(all_json_files)
    print_every = max(1, len(all_json_files) // 100)
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_single_file, path, n_bits): idx for idx, path in enumerate(all_json_files)}
        completed = 0
        for i, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            out_bytes, bits, info = future.result()
            results[idx] = out_bytes
            completed += 1
            if completed % print_every == 0 or completed == len(all_json_files):
                print(f"Processed {completed}/{len(all_json_files)} files...")
    # Concatenate all bits from all samples
    flat_bits = []
    for bits in results:
        if bits is not None:
            flat_bits.extend(bits)
    # Write bits as a true bitstream: pack bits into bytes, in extraction order
    out_bytes = bytearray()
    for i in range(0, len(flat_bits), 8):
        chunk = flat_bits[i:i+8]
        byte = 0
        for j, bit in enumerate(chunk):
            byte |= (bit << j)
        out_bytes.append(byte)
    with open(output_file_path, 'wb') as out_f:
        out_f.write(out_bytes)
    print(f"Done. Output written to {output_file_path}. Total bits: {len(flat_bits)}, total bytes: {len(out_bytes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract n bits from bus data JSON files.")

    parser.add_argument("--n-bits", "-n", type=int, default=8, help="Number of bits to extract (must be even, each bus gives 2 bits).")
    args = parser.parse_args()
    input_folder = 'raw_data/bus'
    output_file = f'processed_data/bus/nbits_{args.n_bits}.bin'
    os.makedirs('processed_data/bus', exist_ok=True)
    process_folder_parallel(input_folder, output_file, args.n_bits)
