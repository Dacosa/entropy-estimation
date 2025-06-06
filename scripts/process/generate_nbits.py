import os
import argparse
import secrets

# Generate n random bits per sample, for num_samples samples, and output as a packed bitstream

def generate_random_nbits_file(output_file, n_bits=8, num_samples=1000000, seed=0xAD5423):
    rng = secrets.SystemRandom(seed)
    total_bits = n_bits * num_samples
    bits = []
    for _ in range(num_samples):
        # For each sample, generate n random bits
        value = rng.getrandbits(n_bits)
        for i in range(n_bits):
            bits.append((value >> i) & 1)  # LSB-first
    # Pack bits into bytes
    out_bytes = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        byte = 0
        for j, bit in enumerate(chunk):
            byte |= (bit << j)
        out_bytes.append(byte)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(out_bytes)
    print(f"Generated {num_samples} samples, {n_bits} bits each ({total_bits} bits, {len(out_bytes)} bytes) to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random n-bit bitstream file.")
    parser.add_argument('--n-bits', '-n', type=int, default=8, help='Number of bits per sample (default: 8)')
    parser.add_argument('--num-samples', '-s', type=int, default=1000000, help='Number of samples (default: 1000000)')
    parser.add_argument('--seed', type=int, default=0xAD5423, help='Random seed (default: 0xAD5423)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file path (default: processed_data/generated/nbits_<n>.bin)')
    args = parser.parse_args()
    output_file = args.output or f'processed_data/generated/nbits_{args.n_bits}.bin'
    generate_random_nbits_file(output_file, n_bits=args.n_bits, num_samples=args.num_samples, seed=args.seed)
