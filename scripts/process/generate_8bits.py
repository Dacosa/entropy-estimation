import os
from pathlib import Path
import secrets

def generate_random_bytes(filepath, num_bytes=10000):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(secrets.token_bytes(num_bytes))
    print(f"Generated {num_bytes} random bytes to {filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a cryptographically secure random 8-bit binary file.")
    parser.add_argument('-n', '--num-bytes', type=int, default=1000000, help='Number of bytes to generate (default: 10000)')
    parser.add_argument('-o', '--output', type=str, default='processed_data/generated/8bits.bin', help='Output file path (default: processed_data/generated/8bits.bin)')
    args = parser.parse_args()
    generate_random_bytes(args.output, args.num_bytes)