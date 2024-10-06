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

# Base URL for the player listings (with placeholders for the index letters)
base_url = 'https://www.sports-reference.com/cbb/players/{}-index.html'

# Initialize an empty list to hold player data
player_data = []

# Iterate through A to Z to cover all last name initials
for letter in string.ascii_lowercase:
    # Construct the URL for each letter
    url = base_url.format(letter)
    print(f"Scraping players with last names starting with {letter.upper()}...")

    # Step 1: Get the HTML content of the page
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')

    player_links = soup.find_all('a', href=re.compile(r'/cbb/players/.*'))

    for link in player_links:
        # Extract player name and remove accents
        player_name = unidecode(link.text)
        player_id = link['href'].split('/')[-1].replace('.html', '')
        
        # Append player data along with the current timestamp
        player_data.append({
            'playerName': player_name,
            'playerId': player_id,
            'timestamp': datetime.now()  # Add timestamp for each entry
        })

# Convert player data to DataFrame
df = pd.DataFrame(player_data)

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
file_name = 'college_data/college_basketball_players.csv'

# Upload the CSV to S3
s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())

print(f"DataFrame uploaded to S3 bucket '{bucket_name}' as '{file_name}'")