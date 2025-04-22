import requests
import os
import time
from datetime import datetime

# Constants
BASE_URL = "http://www.dtpmetropolitano.cl/posiciones"
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "raw_data", "bus")  # Directory to save files
AUTH_FILE = os.path.join(SCRIPT_DIR, "auth.txt")  # File containing username and password
REQUEST_INTERVAL = 60  # 1 minute between requests
MAX_RETRIES = 5  # Maximum number of retries for a failed request
RETRY_DELAY = 5  # Seconds to wait before retrying


def load_auth_credentials(auth_file):
    """Load username and password from a file."""
    try:
        with open(auth_file, "r") as f:
            lines = f.readlines()
            username = lines[0].strip()
            password = lines[1].strip()
            return username, password
    except Exception as e:
        print(f"Error reading auth file: {e}")
        return None, None


def fetch_bus_data(username, password):
    """Fetch bus data from the API and save it."""
    while True:
        now = datetime.now()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Make the request with Basic Authentication
                response = requests.get(BASE_URL, auth=(username, password), timeout=10)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Save the response data
                save_response(response.content, now)
                break  # Exit retry loop if successful

            except requests.RequestException as e:
                print(f"Error fetching bus data (Attempt {attempt}/{MAX_RETRIES}): {e}")

                if attempt < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Skipping this request.")
        else:
            # Log failure after all retries
            log_failure(now)

        # Wait for the next request interval
        print("Waiting for the next request...")
        time.sleep(REQUEST_INTERVAL)


def save_response(data, timestamp):
    """Save the API response to a file in a structured directory."""
    # Create directory for the date
    date_dir = os.path.join(OUTPUT_DIR, timestamp.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    # Save the response as a JSON file with format {hour_min}.json
    filename = os.path.join(date_dir, f"{timestamp.strftime('%H_%M')}.json")
    with open(filename, "wb") as f:
        f.write(data)

    print(f"Saved bus data to {filename}")


def log_failure(timestamp):
    """Log a failure to fetch data."""
    date_dir = os.path.join(OUTPUT_DIR, timestamp.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    # Create a failure log file
    log_file = os.path.join(date_dir, "fetch_failures.log")
    with open(log_file, "a") as f:
        f.write(f"Failed to fetch data at {timestamp.strftime('%H:%M:%S')}\n")

    print(f"Logged failure at {timestamp.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Load credentials
    username, password = load_auth_credentials(AUTH_FILE)
    if not username or not password:
        print("Missing or invalid authentication credentials.")
        exit(1)

    # Start fetching bus data
    fetch_bus_data(username, password)
