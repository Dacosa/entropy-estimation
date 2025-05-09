from cpp_integration.entropy_assessment import run_entropy_assessment

# Replace these with your actual file paths and binary directory
sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
}

method = 'ea_non_iid'
bits_per_symbol = 8

import os

RESULTS_DIR = os.path.join('results', 'nist')
os.makedirs(RESULTS_DIR, exist_ok=True)

for name, input_file in sources.items():
    print(f"\nEstimating entropy for source: {name}")
    try:
        output = run_entropy_assessment(input_file, method=method, bits_per_symbol=bits_per_symbol, verbose=1)
        print(f"Result for {name} ({input_file}):\n{output}")
        result_path = os.path.join(RESULTS_DIR, f"{name}.txt")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(str(output))
        print(f"Saved result to {result_path}")
    except Exception as e:
        print(f"Failed to process {name} ({input_file}): {e}")