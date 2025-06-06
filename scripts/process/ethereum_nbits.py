# Parameterized version of ethereum_8bits.py supporting n bits per output byte.
import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def get_random_positions(seed, num_positions=8, min_pos=0, max_pos=128):
    random.seed(seed)
    return sorted(random.sample(range(min_pos, max_pos), num_positions))

def extract_bits_from_positions(data, selected_positions, n_bits):
    bits = []
    for pos in selected_positions:
        if pos < len(data):
            bits.append(data[pos] & 1)
        else:
            bits.append(0)
    # Truncate or pad to exactly n_bits
    bits = bits[:n_bits] + [0] * max(0, n_bits - len(bits))
    return bits

def process_single_file(json_path, selected_positions, n_bits):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        block_hash = data.get('block_hash', '')
        if isinstance(block_hash, str) and len(block_hash) >= 2:
            hash_hex = block_hash[2:] if block_hash.startswith('0x') else block_hash
            try:
                hash_bytes = bytes.fromhex(hash_hex)
            except Exception:
                hash_bytes = b''
        else:
            hash_bytes = b''
        bits = extract_bits_from_positions(hash_bytes, selected_positions, n_bits)
        return bits, None, None, None
    except Exception as e:
        return None, None, None, f"Error processing {json_path}: {e}"

def process_folder_parallel(input_folder, output_file_path, selected_positions, n_bits):
    input_folder = Path(input_folder)
    all_json_files = sorted(input_folder.rglob('*.json'))
    print(f"Found {len(all_json_files)} JSON files in {input_folder}")
    results = [None] * len(all_json_files)
    print_every = max(1, len(all_json_files) // 100)
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_single_file, path, selected_positions, n_bits): idx for idx, path in enumerate(all_json_files)}
        completed = 0
        for i, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            byte, _, _, info = future.result()
            results[idx] = byte
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
    parser = argparse.ArgumentParser(description="Extract n bits from ethereum data JSON files.")

    parser.add_argument("--n-bits", "-n", type=int, default=8, help="Number of bits to extract per byte (default 8).")
    parser.add_argument("--seed", type=int, default=0xAD5423, help="Random seed for position selection.")
    parser.add_argument("--max-pos", type=int, default=128, help="Maximum position index for sampling.")
    args = parser.parse_args()
    selected_positions = get_random_positions(args.seed, num_positions=args.n_bits, max_pos=args.max_pos)
    print(f"Random positions used: {selected_positions}")
    input_folder = 'raw_data/ethereum'
    output_file = f'processed_data/ethereum/nbits_{args.n_bits}.bin'
    os.makedirs('processed_data/ethereum', exist_ok=True)
    process_folder_parallel(input_folder, output_file, selected_positions, args.n_bits)
