import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
import io
from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging
from datetime import datetime
from unidecode import unidecode
import time

# Base URL for the player listings (for last names starting with a given letter)
base_url = 'https://www.basketball-reference.com/players/{}/'

# Initialize an empty list to hold player data
player_data = []

# Iterate through A to Z to cover all last name initials
for letter in string.ascii_lowercase:
    # Construct the URL for each letter
    url = base_url.format(letter)
    print(f"Scraping players with last names starting with {letter.upper()}...")

    # Step 1: Get the HTML content of the page
    try:
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting for a while...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            response = requests.get(url)  # Retry the request

        if response.status_code != 200:
            print(f"Failed to retrieve data for {letter.upper()}: {response.status_code}")
            continue

        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all player links on the page
        player_links = soup.find_all('a', href=re.compile(r'^/players/.*\.html$'))

        for link in player_links:
            # Get the player name and ID
            player_name = unidecode(link.text.strip())  # Remove accents
            if player_name and player_name != 'Players':
                player_id = link['href'].split('/')[-1].replace('.html', '')
                player_data.append({
                    'playerName': player_name, 
                    'playerId': player_id,
                    'timestamp': datetime.now()  # Add timestamp
                })

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Wait a bit before the next request to avoid rate limiting
    time.sleep(2)  # Adjust the delay as needed

# Create a DataFrame and remove duplicates
df = pd.DataFrame(player_data).drop_duplicates()

# Remove any unwanted rows with 'Players' as a name
df = df[df['playerName'] != 'Players']

# Load AWS credentials from environment variables
load_dotenv()

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION')

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

# Convert DataFrame to CSV and store it in memory
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)

# Define your S3 bucket and file name
bucket_name = 'hooperdna-storage'
file_name = 'nba_player_data/nba_players_n_ids.csv'

# Upload the CSV to S3
s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())

print(f"DataFrame uploaded to S3 bucket '{bucket_name}' as '{file_name}'")