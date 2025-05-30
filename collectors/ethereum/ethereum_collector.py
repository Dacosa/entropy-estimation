import requests
import json
import os
from datetime import datetime
import time

# Constants
API_URL = "https://api.etherscan.io/api"
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "raw_data", "ethereum")  # Directory to save files
MAX_CALLS_PER_SECOND = 4
BLOCK_FETCH_COUNT = 10000
AUTH_FILE = os.path.join(SCRIPT_DIR, "auth.txt")  # File containing API key

# Read API key from auth.txt in the same folder as this script
with open(AUTH_FILE, 'r') as f:
    API_KEY = f.read().strip()


def get_block_data(block_number: str) -> dict:
    """
    Fetch Ethereum block data by block number using the Etherscan API.
    """
    params = {
        "module": "proxy",
        "action": "eth_getBlockByNumber",
        "tag": block_number,
        "boolean": "true",  # Include full transaction objects
        "apikey": API_KEY,
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if "result" in data:
            return data["result"]
    return None


def save_block_data(block_hash: str, block_number: str, timestamp: int):
    """
    Save block data in the specified directory format.
    """
    dir_path = OUTPUT_DIR
    os.makedirs(dir_path, exist_ok=True)
    # Format the timestamp to the minute
    dt = datetime.utcfromtimestamp(timestamp)
    file_name = dt.strftime("%Y%m%d_%H%M.json")
    file_path = os.path.join(dir_path, file_name)
    block_data = {
        "block_hash": block_hash,
        "block_number": block_number,
        "timestamp": timestamp
    }
    # Write data to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(block_data, f, indent=4)
    print(f"Block data saved: {file_path}")


def collect_blocks_backwards(start_block=None, n_blocks=BLOCK_FETCH_COUNT):
    """
    Collect n_blocks Ethereum blocks backwards from start_block (inclusive).
    If start_block is None, start from the latest block.
    start_block can be an int or hex string.
    """
    if start_block is None:
        latest_block_data = get_block_data("latest")
        if not latest_block_data:
            print("Failed to fetch the latest block data.")
            return
        start_block_number = int(latest_block_data["number"], 16)
    else:
        if isinstance(start_block, str):
            if start_block.startswith("0x"):
                start_block_number = int(start_block, 16)
            else:
                start_block_number = int(start_block)
        else:
            start_block_number = int(start_block)

    for i in range(n_blocks):
        block_number_hex = hex(start_block_number - i)
        block_data = get_block_data(block_number_hex)
        if block_data:
            block_hash = block_data["hash"]
            block_number = block_data["number"]
            timestamp = int(block_data["timestamp"], 16)
            save_block_data(block_hash, block_number, timestamp)
        else:
            print(f"Failed to fetch data for block {block_number_hex}")
        if (i + 1) % MAX_CALLS_PER_SECOND == 0:
            time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-block', type=str, default=None, help='Block number (int or hex) to start collecting from (inclusive). If not given, starts from latest.')
    parser.add_argument('--n-blocks', type=int, default=BLOCK_FETCH_COUNT, help='Number of blocks to collect (default: 10000)')
    args = parser.parse_args()
    collect_blocks_backwards(start_block=args.start_block, n_blocks=args.n_blocks)


if __name__ == "__main__":
    collect_last_n_blocks()
