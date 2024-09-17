import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

year = 2024

# URL for the selected year's season's data
url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html#per_game_stats::pts_per_g'

# Directory for saving CSV files
raw_save_dir = "./sample_DB/nba_raw_data"
clean_save_dir = "./sample_DB/nba_clean_data"

# Ensure the directories exist
os.makedirs(raw_save_dir, exist_ok=True)
os.makedirs(clean_save_dir, exist_ok=True)

# Mapping of old column names to camel case with capital starting letters and no spaces
column_mapping = {
    'Rk': 'Rank', 'Player': 'Player', 'Age': 'Age', 'Tm': 'Team', 'Pos': 'Position',
    'G': 'Games', 'GS': 'GamesStarted', 'MP': 'MinutesPlayed', 'FG': 'FieldGoalsMade',
    'FGA': 'FieldGoalsAttempted', 'FG%': 'FieldGoalPercentage', '3P': 'ThreePointFieldGoalsMade',
    '3PA': 'ThreePointFieldGoalsAttempted', '3P%': 'ThreePointFieldGoalPercentage',
    '2P': 'TwoPointFieldGoalsMade', '2PA': 'TwoPointFieldGoalsAttempted', '2P%': 'TwoPointFieldGoalPercentage',
    'eFG%': 'EffectiveFieldGoalPercentage', 'FT': 'FreeThrowsMade', 'FTA': 'FreeThrowsAttempted',
    'FT%': 'FreeThrowPercentage', 'ORB': 'OffensiveRebounds', 'DRB': 'DefensiveRebounds', 'TRB': 'TotalRebounds',
    'AST': 'Assists', 'STL': 'Steals', 'BLK': 'Blocks', 'TOV': 'Turnovers', 'PF': 'PersonalFouls', 'PTS': 'Points',
    'Awards': 'Awards'
}

# Raw and clean CSV file paths
raw_csv_file = os.path.join(raw_save_dir, f'{year}_NBAPlayerStats_HprDNA_raw.csv')
clean_csv_file = os.path.join(clean_save_dir, f'{year}_NBAPlayerStats_HprDNA_cln.csv')

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
    season = f"{year-1}-{year-2000}"
    top_half_df.insert(0, 'Season', season)
    
    # Save the raw data into the raw directory
    top_half_df.to_csv(raw_csv_file, index=False)
    
    # Rename the columns based on the camel case column mapping
    revised_df = top_half_df.rename(columns=column_mapping)
    
    # Save the cleaned data into the clean directory
    revised_df.to_csv(clean_csv_file, index=False)
    
    print(f"Data for the {year} season has been saved: Raw ({raw_csv_file}), Clean ({clean_csv_file}).")
else:
    print(f"Table with the specified ID not found for the {year} season.")
