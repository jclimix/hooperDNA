import duckdb
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

college_player_id = 'darius-johnson-3'

# College player dictionary with all values set to 0 initially
college_player = {
    'G': 0,       # Games played
    'GS': 0,      # Games started
    'MP': 0.0,    # Minutes per game
    'FG': 0.0,    # Field Goals Made per game
    'FGA': 0.0,   # Field Goals Attempted per game
    'FG%': 0.0,   # Field Goal Percentage
    '3P': 0.0,    # Three-Point Field Goals Made per game
    '3PA': 0.0,   # Three-Point Field Goals Attempted per game
    'FT': 0.0,    # Free Throws Made per game
    'FTA': 0.0,   # Free Throws Attempted per game
    'FT%': 0.0,   # Free Throw Percentage
    'ORB': 0.0,   # Offensive Rebounds per game
    'DRB': 0.0,   # Defensive Rebounds per game
    'TRB': 0.0,   # Total Rebounds per game
    'AST': 0.0,   # Assists per game
    'STL': 0.0,   # Steals per game
    'BLK': 0.0,   # Blocks per game
    'TOV': 0.0,   # Turnovers per game
    'PF': 0.0,    # Personal Fouls per game
    'PTS': 0.0    # Points per game
}

# Web scraping the college player's stats
url = f'https://www.sports-reference.com/cbb/players/{college_player_id}.html'
response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

div = soup.find('div', id='div_players_per_game')

if div:
    table = div.find('table')

    if table:
        college_player_stats_df = pd.read_html(str(table))[0]

        # Display the DataFrame for verification
        print("College Player Stats DataFrame:")
        print(college_player_stats_df)

        # Extract the second to last row and update the college_player dictionary
        if not college_player_stats_df.empty:
            latest_stats = college_player_stats_df.iloc[-2]  # Extract second to last row
            for stat in college_player.keys():
                if stat in latest_stats.index:
                    college_player[stat] = latest_stats[stat]
        
        # Display the selected row for cosine similarity analysis
        print("\nSelected College Player Stats for Cosine Similarity:")
        print(pd.DataFrame([latest_stats], columns=college_player_stats_df.columns))
    else:
        print("Table not found on the page.")
else:
    print("Div with player stats not found on the page.")

# Convert updated college player's stats into a NumPy array for cosine similarity
college_stats = np.array(list(college_player.values())).reshape(1, -1)

# Initialize an empty list to store DataFrames
results = []

# Directory where CSV files are located
directory = './sample_DB/'

# Loop through CSVs from 2015 to 2024
for year in range(2015, 2025):
    # Define the file name for the current season
    csv_file = os.path.join(directory, f'{year}NBAPlayerStats_HprDNA.csv')
    
    if os.path.exists(csv_file):
        # Load the data from CSV using DuckDB
        query = f"SELECT * FROM '{csv_file}'"
        season_data = duckdb.query(query).df()

        # Filter out only the stats that are present in the college_player dictionary for comparison
        stat_columns = list(college_player.keys())
        
        if all(col in season_data.columns for col in stat_columns):
            nba_stats = season_data[stat_columns]

            # Replace missing values with 0 to avoid issues with similarity calculation
            nba_stats = nba_stats.fillna(0)

            # Extract the position of the college player
            college_player_pos = latest_stats.get('Pos', 'Unknown')

            # Map college positions to NBA positions
            pos_mapping = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C']}
            nba_positions = pos_mapping.get(college_player_pos, [])

            if nba_positions:
                # Filter NBA players by position
                filtered_season_data = season_data[season_data['Pos'].isin(nba_positions)]

                # Filter out only the stats that are present in the college_player dictionary for comparison
                filtered_nba_stats = filtered_season_data[stat_columns]
                filtered_nba_stats = filtered_nba_stats.fillna(0)

                # Calculate cosine similarity between college player and filtered NBA players in the current season
                similarities = cosine_similarity(filtered_nba_stats, college_stats)

                # Find the NBA player with the highest similarity
                max_sim_index = np.argmax(similarities)
                most_similar_player = filtered_season_data.iloc[max_sim_index]

                # Convert the most_similar_player to DataFrame with a single row
                most_similar_player_df = pd.DataFrame([most_similar_player], columns=filtered_season_data.columns)

                # Convert the similarity score to a percentage
                similarity_percentage = similarities[max_sim_index][0] * 100

                # Ensure that the DataFrame only contains one row of data
                most_similar_player_df = most_similar_player_df.head(1)

                # Add the similarity score to the DataFrame
                most_similar_player_df.loc[most_similar_player_df.index[0], 'Similarity (%)'] = f'{similarity_percentage:.2f}%'

                # Append the DataFrame to the list
                results.append(most_similar_player_df)
            else:
                print(f"No matching NBA positions found for college position: {college_player_pos}.")
        else:
            print(f"One or more columns from {stat_columns} are missing in {csv_file}.")
    else:
        print(f"File not found: {csv_file}")

# Concatenate all results into a single DataFrame
most_similar_players = pd.concat(results, ignore_index=True)

# Drop rows with all null values
most_similar_players = most_similar_players.dropna(how='all')

# Display the resulting DataFrame
print("\nMost Similar NBA Players:")
print(most_similar_players)
