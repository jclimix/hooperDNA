import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string

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

    # Step 2: Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 3: Find all the links to player profiles
    player_links = soup.find_all('a', href=re.compile(r'/cbb/players/.*'))

    # Step 4: Extract player names and their IDs from the links
    for link in player_links:
        player_name = link.text
        player_id = link['href'].split('/')[-1].replace('.html', '')
        player_data.append({'Player Name': player_name, 'Player ID': player_id})

# Step 5: Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(player_data)

# Step 6: Save the DataFrame to a CSV file
df.to_csv('college_basketball_players.csv', index=False)

print("Scraping complete! Data saved to 'college_basketball_players.csv'.")
