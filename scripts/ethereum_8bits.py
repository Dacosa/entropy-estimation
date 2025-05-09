import os
import json
from pathlib import Path
from datetime import datetime

# Input/output paths
INPUT_DIR = os.path.join('raw_data', 'ethereum')
OUTPUT_DIR = os.path.join('processed_data', 'ethereum')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, '8bits.bin')

# Helper: get last 2 bits of an integer
get_2bits = lambda x: int(x) & 0b11

def process_ethereum_blocks(input_dir, output_file):
    files = sorted(Path(input_dir).glob('*.json'))
    print(f"Found {len(files)} block files in {input_dir}")
    output_bytes = bytearray()
    for idx, f in enumerate(files):
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            block_hash = data.get('block_hash', '')
            # Extract first 8 bits from the hash (skip '0x' if present)
            if isinstance(block_hash, str):
                hash_hex = block_hash[2:] if block_hash.startswith('0x') else block_hash
                hash_hex = hash_hex[:2]  # first byte (2 hex chars)
                try:
                    byte_val = int(hash_hex, 16)
                except Exception:
                    byte_val = 0
            else:
                byte_val = 0
            output_bytes.append(byte_val)
        except Exception as e:
            print(f"Error processing {f}: {e}")
        if idx % (max(1, len(files)//100)) == 0:
            print(f"Progress: {idx * 100 // max(1, len(files))}% ({idx}/{len(files)})")
    with open(output_file, 'wb') as out_f:
        out_f.write(output_bytes)
    print(f"Done. Wrote {len(output_bytes)} bytes to {output_file}")

if __name__ == "__main__":
    process_ethereum_blocks(INPUT_DIR, OUTPUT_FILE)
