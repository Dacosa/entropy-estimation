import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

SEED_HEX = 'ad5423'
SEED_INT = int(SEED_HEX, 16)

# Generate 8 random positions in [1, 99840]
SELECTED_POSITIONS = None

def get_random_positions(seed, num_positions=8, min_position=1, max_position=99840):
    random.seed(seed)
    return sorted(random.sample(range(min_position, max_position + 1), num_positions))

def extract_byte_from_positions(data, selected_positions):
    bits = []
    for pos in selected_positions:
        if pos < len(data):
            bits.append((data[pos] & 1))
        else:
            bits.append(0)
    byte = 0
    for i, bit in enumerate(bits):
        byte |= (bit << (7 - i))
    return bytes([byte])

def process_radio_file(file_path, selected_positions):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        num_minutes = len(data) // 960_000  # 128kbps = 960,000 bytes per minute
        byte_list = []
        for minute in range(num_minutes):
            extracted = extract_byte_from_positions(data, selected_positions)
            if extracted:
                byte_list.append(extracted)
        return b''.join(byte_list), len(byte_list)
    except Exception as e:
        return None, f"Error processing {file_path}: {e}"

def process_radio_files(input_dir, selected_positions, output_path):
    input_dir = Path(input_dir)
    radio_files = sorted([p for p in input_dir.rglob('*') if p.is_file()])
    print(f"Found {len(radio_files)} files in {input_dir}")
    total_files = len(radio_files)
    all_bytes = []
    start_time = time.time()
    last_printed_percent = -1
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_radio_file, path, selected_positions): path
            for path in radio_files
        }
        completed = 0
        for future in as_completed(future_to_file):
            result = future.result()
            completed += 1
            percent = int((completed / total_files) * 100)
            if percent > last_printed_percent:
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time_per_file = elapsed / completed
                    est_total = avg_time_per_file * total_files
                    est_remaining = est_total - elapsed
                    print(f"Progress: {percent}% ({completed}/{total_files}) | Elapsed: {elapsed:.1f}s | Est. remaining: {est_remaining:.1f}s")
                last_printed_percent = percent
            if result and result[0]:
                all_bytes.append(result[0])
            elif result:
                print(result[1])
    print("Done.")
    path = Path(output_path)
    with open(path, 'wb') as f:
        for b in all_bytes:
            f.write(b)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    input_dir = 'raw_data/radio'
    output_dir = 'processed_data/radio'
    SELECTED_POSITIONS = get_random_positions(SEED_INT)
    print(f"Random positions used: {SELECTED_POSITIONS}")
    output_path = os.path.join(output_dir, '8bits.bin')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    process_radio_files(input_dir, SELECTED_POSITIONS, output_path)