import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
from unidecode import unidecode  # Import unidecode to handle accented characters

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
        # Get player name and convert any accented characters to their unaccented counterparts
        player_name = unidecode(link.text)
        player_id = link['href'].split('/')[-1].replace('.html', '')
        player_data.append({'playerName': player_name, 'playerId': player_id})

# Create DataFrame and remove rows with the header 'Players'
df = pd.DataFrame(player_data)
df = df[df['playerName'] != 'Players']

# Save DataFrame to CSV
df.to_csv('./sample_DB/college_data/college_basketball_players.csv', index=False)

print("Scraping complete! Data saved to 'college_basketball_players.csv'.")
