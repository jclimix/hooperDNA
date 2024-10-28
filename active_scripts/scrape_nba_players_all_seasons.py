import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

# Base URL for scraping
base_url = 'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'

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

# Initial year
start_year = 2011
end_year = 2014  # You can adjust this as needed

# Loop over the years
for year in range(start_year, end_year + 1):
    # Format the URL with the current year
    url = base_url.format(year=year)
    success = False
    retries = 3
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        
        # Check for successful response
        if response.status_code == 200:
            success = True
            break
        elif response.status_code == 429:
            print(f"Rate limit hit for year {year}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
        else:
            print(f"Failed to retrieve the webpage for year {year}. Status code: {response.status_code}")
            break
    
    if not success:
        print(f"Skipping year {year} due to repeated errors.")
        continue

    # Store HTML content in memory
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    
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
    
    # Delay to avoid hitting rate limits
    time.sleep(2)  # Adjust delay as needed
