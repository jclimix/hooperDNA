import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

# Base URL for scraping
base_url = 'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html#per_game_stats::pts_per_g'

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
        
        # Take the top half of the DataFrame
        top_half_df = df.head(len(df) // 2)
        
        # Add 'Season' column
        season = f"{year-1}-{str(year)[-2:]}"
        top_half_df.insert(0, 'Season', season)
        
        # Save each season's data into its own CSV file
        csv_file = f"./sample_DB/{year}NBAPlayerStats_HprDNA.csv"
        top_half_df.to_csv(csv_file, index=False)
        
        print(f"Data for year {year} saved to {csv_file} successfully.")
    else:
        print(f"Table with the specified ID not found for year {year}.")
