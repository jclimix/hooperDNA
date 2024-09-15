import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

# URL for the 2024 season data
url = 'https://www.basketball-reference.com/leagues/NBA_2024_per_game.html#per_game_stats::pts_per_g'

# CSV file to update
csv_file = f'./sample_DB/2024NBAPlayerStats_HprDNA.csv'

# Send an HTTP request to the URL with proper encoding
response = requests.get(url)
response.encoding = 'utf-8'  # Ensure the response is interpreted as UTF-8

# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the specified ID
table = soup.find('table', id='per_game_stats')

if table:
    # Use pandas to read the table HTML into a DataFrame
    df = pd.read_html(str(table))[0]
    
    # Remove accent marks from the 'Player' column
    if 'Player' in df.columns:
        df['Player'] = df['Player'].apply(lambda x: unidecode(str(x)))
    
    # Take the top half of the DataFrame
    top_half_df = df.head(len(df) // 2)
    
    # Add 'Season' column
    season = f"2023-24"
    top_half_df.insert(0, 'Season', season)
    
    # Overwrite the CSV file with updated 2024 data
    top_half_df.to_csv(csv_file, index=False)
    
    print(f"Data for the 2024 season has been updated and saved to {csv_file}.")
else:
    print(f"Table with the specified ID not found for the 2024 season.")
