import requests
import os
import time
from datetime import datetime

# Constants
STREAM_URL = "https://sonic-us.streaming-chile.com/8186/stream"  # Radio UChile
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "raw_data", "radio")  # Base directory for saved files
CHUNK_SIZE = 1024 * 1024 * 10  # ~10 MB file size
INITIAL_RETRY_DELAY = 5  # Initial retry delay in seconds
MAX_RETRY_DELAY = 300  # Maximum retry delay in seconds


def fetch_audio_data():
    """Fetch radio stream data and save it in the specified format."""
    retry_delay = INITIAL_RETRY_DELAY
    while True:  # Run indefinitely
        try:
            print("Connecting to the radio stream...")
            response = requests.get(STREAM_URL, stream=True, timeout=10)
            response.raise_for_status()
            print("Connection established.")
            process_stream(response)
            retry_delay = INITIAL_RETRY_DELAY  # Reset retry delay on success

        except requests.RequestException as e:
            print(f"Error connecting to stream: {e}")
            log_failure(f"Error connecting to stream: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)  # Exponential backoff

        except Exception as e:
            print(f"Unexpected error: {e}")
            log_failure(f"Unexpected error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)


def process_stream(response):
    """Process the radio stream and save chunks to files."""
    current_file = None
    current_size = 0

    try:
        for chunk in response.iter_content(chunk_size=1024):  # 1 KB fetch chunks
            if not chunk:  # Handle unexpected empty chunks
                print("Received an empty chunk, skipping...")
                continue

            if current_file is None:
                current_file, current_size = open_new_file()

            # Write chunk to the current file
            current_file.write(chunk)
            current_size += len(chunk)

            # Rotate file if size exceeds CHUNK_SIZE
            if current_size >= CHUNK_SIZE:
                current_file.close()
                print(f"Closed file after writing {current_size} bytes")
                current_file = None  # Trigger a new file in the next iteration

    except Exception as e:
        print(f"Error during stream processing: {e}")
        log_failure(f"Stream processing error: {e}")
        raise  # Re-raise the exception to trigger a retry in fetch_audio_data()

    finally:
        if current_file:
            current_file.close()
            print(f"Closed file after writing {current_size} bytes")


def open_new_file():
    """Create a new file with a timestamped name."""
    now = datetime.now()
    date_dir = os.path.join(OUTPUT_DIR, now.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    # File name format: {hour_min_sec}.mp3
    filename = os.path.join(date_dir, f"{now.strftime('%H_%M_%S')}.mp3")
    file = open(filename, "wb")
    print(f"Opened new file: {filename}")
    log_success(f"Opened new file: {filename}")
    return file, 0  # Initialize size to 0


def log_success(message):
    """Log a success message."""
    log_file = os.path.join(OUTPUT_DIR, "collector.log")
    with open(log_file, "a") as log:
        log.write(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def log_failure(message):
    """Log a failure message."""
    log_file = os.path.join(OUTPUT_DIR, "collector.log")
    with open(log_file, "a") as log:
        log.write(f"[FAILURE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


if __name__ == "__main__":
    fetch_audio_data()
