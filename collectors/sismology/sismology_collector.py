import requests
from bs4 import BeautifulSoup
import pandas as pd  # pandas is required for Parquet
import time
import os

# Base URL of the page to scrape
base_url = 'https://evtdb.csn.uchile.cl/?page='
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "raw_data", "sismology")  # Directory to save files
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sismology_data.parquet")

# Prepare a list to collect all rows
records = []

# Loop through all pages (adjust the range as needed)
for page in range(1, 194):
    print(f"Scraping page {page}...")
    # Send a request to fetch the page content
    response = requests.get(base_url + str(page))
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all rows in the table's body, skipping headers
    rows = soup.find_all('tr')[8:]  # Skip the first rows that are headers
    # Extract data from each row
    for row in rows:
        columns = row.find_all('td')
        if len(columns) == 5:  # Ensure there are 5 columns
            date = columns[0].text.strip()
            latitude = columns[1].text.strip()
            longitude = columns[2].text.strip()
            depth = columns[3].text.strip()
            magnitude = columns[4].text.strip()
            records.append({
                'Date': date,
                'Latitude': latitude,
                'Longitude': longitude,
                'Depth': depth,
                'Magnitude': magnitude
            })
    # Wait for a few seconds to avoid overloading the server
    time.sleep(2)

# Convert the records to a DataFrame and save as Parquet
if records:
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Data has been extracted and saved to '{OUTPUT_FILE}'.")
else:
    print("No data found to save.")
