import json
from pathlib import Path
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_bus_file(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data

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
    # Handles both 'YYYYMMDDHHMMSS' and 'DD-MM-YYYY HH:MM:SS'
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except ValueError:
        return datetime.strptime(dt_str, "%d-%m-%Y %H:%M:%S")

def extract_last_decimal_digit(coord_str):
    try:
        # Get the last digit of the decimal part
        if '.' in coord_str:
            return int(coord_str.strip().split('.')[-1][-1])
        else:
            return 0
    except Exception:
        return 0

def process_single_file(json_path):
    try:
        data = load_bus_file(json_path)
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
        active_buses = [
            (bus_id, info[0], info[1], info[2])
            for bus_id, info in bus_latest.items()
            if request_time - info[0] <= timedelta(minutes=2)
        ]
        active_buses.sort(key=lambda x: x[1], reverse=True)
        digits = []
        for bus in active_buses[:4]:
            lat_digit = extract_last_decimal_digit(bus[2])
            lon_digit = extract_last_decimal_digit(bus[3])
            digits.append(lat_digit)
            digits.append(lon_digit)
        while len(digits) < 8:
            digits.append(0)
        bits = [(d % 2) for d in digits]
        byte = 0
        for i, bit in enumerate(bits):
            byte |= (bit << (7 - i))
        return byte, digits, bits, str(json_path)
    except Exception as e:
        return None, None, None, f"Error processing {json_path}: {e}"

def process_folder_parallel(input_folder, output_file_path):
    input_folder = Path(input_folder)
    all_json_files = sorted(input_folder.rglob('*.json'))
    print(f"Found {len(all_json_files)} JSON files in {input_folder}")
    results = [None] * len(all_json_files)
    print_every = max(1, len(all_json_files) // 100)  # Print progress every 1% or at least every file if <100
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_single_file, path): idx for idx, path in enumerate(all_json_files)}
        completed = 0
        for i, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            byte, digits, bits, info = future.result()
            results[idx] = (byte, digits, bits, info)
            completed += 1
            if completed % print_every == 0 or completed == len(all_json_files):
                print(f"Processed {completed}/{len(all_json_files)} files...")
    # Write all bytes in the original order
    with open(output_file_path, 'wb') as out_f:
        for res in results:
            if res and res[0] is not None:
                out_f.write(bytes([res[0]]))
    print("Done. Output written to", output_file_path)

if __name__ == "__main__":
    input_folder = os.path.join('raw_data', 'bus', '2024-11-27')
    output_file = os.path.join('processed_data', 'bus', '8bits')
    process_folder_parallel(input_folder, output_file)
