import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
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
            player_name = link.text.strip()
            if player_name and player_name != 'Players':
                player_id = link['href'].split('/')[-1].replace('.html', '')
                player_data.append({'playerName': player_name, 'playerId': player_id})

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Wait a bit before the next request to avoid rate limiting
    time.sleep(2)  # Adjust the delay as needed

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(player_data)

# Remove any potential duplicate entries
df = df.drop_duplicates()

# Save the DataFrame to a CSV file
df.to_csv('./sample_DB/nba_player_data/nba_players_n_ids.csv', index=False)

print("Scraping complete! Data saved to 'nba_players_n_ids.csv'.")
