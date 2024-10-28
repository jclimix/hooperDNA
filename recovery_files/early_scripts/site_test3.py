import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

college_player_id = 'jaedon-ledee-1'

# player stats dictionary (college)
college_player = {
    'MP': 0.0,    # Minutes per game
    'FG': 0.0,    # Field Goals Made per game
    'FGA': 0.0,   # Field Goals Attempted per game
    'FG%': 0.0,   # Field Goal Percentage
    '3P': 0.0,    # Three-Point Field Goals Made per game
    '3PA': 0.0,   # Three-Point Field Goals Attempted per game
    '3P%': 0.0,   # Three-Point Field Goals Attempted per game
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

# initial raw weights
raw_weights = {
    'MP': 6.0,    # Minutes per game
    'FG': 5.0,    # Field Goals Made per game
    'FGA': 5.0,   # Field Goals Attempted per game
    'FG%': 8.0,   # Field Goal Percentage
    '3P': 5.0,    # Three-Point Field Goals Made per game
    '3PA': 5.0,   # Three-Point Field Goals Attempted per game
    '3P%': 8.0,   # Three-Point Field Goals Attempted per game
    'FT': 5.0,    # Free Throws Made per game
    'FTA': 5.0,   # Free Throws Attempted per game
    'FT%': 5.0,   # Free Throw Percentage
    'ORB': 5.0,   # Offensive Rebounds per game
    'DRB': 5.0,   # Defensive Rebounds per game
    'TRB': 5.0,   # Total Rebounds per game
    'AST': 5.0,   # Assists per game
    'STL': 5.0,   # Steals per game
    'BLK': 5.0,   # Blocks per game
    'TOV': 4.0,   # Turnovers per game
    'PF': 4.0,    # Personal Fouls per game
    'PTS': 5.0    # Points per game
}

# normalize weights so they sum to 100% (1.0)
total_weight = sum(raw_weights.values())
weights = {stat: value / total_weight for stat, value in raw_weights.items()}

# Scrape College Basketball Ref site to pull selected player's stats

url = f'https://www.sports-reference.com/cbb/players/{college_player_id}.html'
response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

div = soup.find('div', id='div_players_per_game')

if div:
    table = div.find('table')

    if table:
        college_player_stats_df = pd.read_html(str(table))[0]

        print("College Player Stats:")
        print(college_player_stats_df)

        # if the 'Season' column exists
        if 'Season' in college_player_stats_df.columns:

            # find index of 'Career' row
            career_index = college_player_stats_df[college_player_stats_df['Season'] == 'Career'].index

            if not career_index.empty:

                # pull the row before 'Career' row
                latest_stats_index = career_index[0] - 1

                # check index
                if latest_stats_index >= 0:
                    latest_stats = college_player_stats_df.iloc[latest_stats_index]

                    # update the college_player dict with latest stats
                    for stat in college_player.keys():
                        if stat in latest_stats.index:
                            college_player[stat] = latest_stats[stat]

                    print("\nSelected College Player Statline for Euclidean Distance Analysis:")
                    print(pd.DataFrame([latest_stats], columns=college_player_stats_df.columns))

                    # college stat adjustments (NCAA => NBA)
                    college_player["MP"] *= 1.2 #40 vs 48 total min
                    college_player["PTS"] *= 1.35 #skew scoring for better offensive player matches

                else:
                    print("No valid row found before 'Career' row.")
            else:
                print("'Career' row not found in the stats table.")
        else:
            print("'Season' column not found in the stats table.")
    else:
        print("Table not found on the page.")
else:
    print("Div with player stats not found on the page.")


# Convert updated college player's stats into a weighted NumPy array for Euclidean distance
college_stats = np.array([college_player[stat] * weights[stat] for stat in college_player.keys()]).reshape(1, -1)

# Empty list to store DataFrames
results = []

dir = './sample_DB/'

# 2015 to 2024
for year in range(2015, 2025):

    # Define the file name for the current season
    csv_file = os.path.join(dir, f'{year}NBAPlayerStats_HprDNA.csv')
    
    if os.path.exists(csv_file):
        # DuckDB: Load data from CSV
        query = f"SELECT * FROM '{csv_file}'"
        season_data = duckdb.query(query).df()

        # Filter out only the stats that are present in the college_player dict for comparison
        stat_columns = list(college_player.keys())
        
        if all(col in season_data.columns for col in stat_columns):
            nba_stats = season_data[stat_columns]

            # Replace missing values with 0 (or else errors)
            nba_stats = nba_stats.fillna(0)

            # Pull position of the college player
            college_player_pos = latest_stats.get('Pos', 'Unknown')

            pos_mapping = {'G': ['PG', 'SG'], 
                           'F': ['SF', 'PF'], 
                           'C': ['C']}
            
            nba_positions = pos_mapping.get(college_player_pos, [])

            if nba_positions:
                # Filter NBA players by position
                filtered_season_data = season_data[season_data['Pos'].isin(nba_positions)].reset_index(drop=True)

                # Filter out only the stats that are present in the college_player dictionary for comparison
                filtered_nba_stats = filtered_season_data[stat_columns]
                filtered_nba_stats = filtered_nba_stats.fillna(0)

                # Apply the weights to NBA stats for comparison
                weighted_nba_stats = filtered_nba_stats.apply(lambda row: row * np.array([weights[stat] for stat in stat_columns]), axis=1)

                # Calculate Euclidean distance between weighted college player stats and weighted NBA players' stats
                distances = weighted_nba_stats.apply(lambda row: euclidean(row, college_stats.flatten()), axis=1)

                # Convert distances Series to DataFrame with proper index
                distances_df = pd.DataFrame({'Distance': distances.values}, index=filtered_season_data.index)

                # Find the NBA player with the smallest distance
                min_dist_index = np.argmin(distances)

                # show distances and index (testing)
                # print("Distances DataFrame:")
                # print(distances_df)
                # print("Index with minimum distance:", min_dist_index)

                # Check if the index exists
                if min_dist_index in distances_df.index:
                    # Convert the distance to a percentage (optional, for easier interpretation)
                    distance_percentage = (1 / (1 + distances_df.loc[min_dist_index, 'Distance'])) * 100

                    # Ensure that the DataFrame only contains one row of data
                    most_similar_player_df = pd.DataFrame([filtered_season_data.iloc[min_dist_index]], columns=filtered_season_data.columns)

                    # Add the distance score to the DataFrame
                    most_similar_player_df.loc[most_similar_player_df.index[0], 'Similarity (%)'] = f'{distance_percentage:.2f}%'

                    # Append the DataFrame to the list
                    results.append(most_similar_player_df)
                else:
                    print(f"Index {min_dist_index} is not found in distances DataFrame.")
            else:
                print(f"No matching NBA positions found for college position: {college_player_pos}.")
        else:
            print(f"One or more columns from {stat_columns} are missing in {csv_file}.")
    else:
        print(f"File not found: {csv_file}")

player_dna_results_df = pd.concat(results, ignore_index=True)

# drop rows with ALL null values (or else error)
most_similar_players = player_dna_results_df.dropna(how='all')

player_dna_results_df = player_dna_results_df.sort_values("Similarity (%)", ascending=False)

print("\nNBA Player 'DNA' Matches:")
print(player_dna_results_df)
