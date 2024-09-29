import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

# Base URL for scraping
base_url = 'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html#per_game_stats::pts_per_g'

# Directory for saving CSV files
raw_save_dir = "./sample_DB/nba_raw_data"
clean_save_dir = "./sample_DB/nba_clean_data"

# Ensure the directories exist
os.makedirs(raw_save_dir, exist_ok=True)
os.makedirs(clean_save_dir, exist_ok=True)

# Mapping of old column names to camel case with capital starting letters and no spaces
column_mapping = {
    'Rk': 'rank', 'Player': 'player', 'Age': 'age', 'Tm': 'team', 'Pos': 'position',
    'G': 'games', 'GS': 'games_started', 'MP': 'minutes_played', 'FG': 'field_goals_made',
    'FGA': 'field_goals_attempted', 'FG%': 'field_goal_percentage', '3P': 'three_point_field_goals_made',
    '3PA': 'three_point_field_goals_attempted', '3P%': 'three_point_field_goal_percentage',
    '2P': 'two_point_field_goals_made', '2PA': 'two_point_field_goals_attempted', '2P%': 'two_point_field_goal_percentage',
    'eFG%': 'effective_field_goal_percentage', 'FT': 'free_throws_made', 'FTA': 'free_throws_attempted',
    'FT%': 'free_throw_percentage', 'ORB': 'offensive_rebounds', 'DRB': 'defensive_rebounds', 'TRB': 'total_rebounds',
    'AST': 'assists', 'STL': 'steals', 'BLK': 'blocks', 'TOV': 'turnovers', 'PF': 'personal_fouls', 'PTS': 'points',
    'Awards': 'awards'
}

# Initial year
start_year = 2015
end_year = 2024  # You can adjust this as needed

# Loop over the years
for year in range(start_year, end_year + 1):
    # Format the URL with the current year
    url = base_url.format(year=year)
    
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
        
        # Add 'Season' column
        season = f"{year-1}-{str(year)[-2:]}"
        df.insert(0, 'Season', season)
        
        # Save the raw data into its own CSV file
        raw_csv_file = os.path.join(raw_save_dir, f"{year}_NBAPlayerStats_HprDNA_raw.csv")
        df.to_csv(raw_csv_file, index=False)
        
        # Rename the columns based on the camel case column mapping
        revised_df = df.rename(columns=column_mapping)
        
        # Save the revised data with the camel case column names into a separate CSV file
        clean_csv_file = os.path.join(clean_save_dir, f"{year}_NBAPlayerStats_HprDNA_cln.csv")
        revised_df.to_csv(clean_csv_file, index=False)
        
        print(f"Data for year {year} saved: Raw ({raw_csv_file}), Clean ({clean_csv_file})")
    else:
        print(f"Table with the specified ID not found for year {year}.")
