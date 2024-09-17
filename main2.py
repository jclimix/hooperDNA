import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode

# Column mapping from original script
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

college_player_id = 'caitlin-clark-1'

# Player stats dictionary (college)
college_player = {
    'MP': 0.0, 'FG': 0.0, 'FGA': 0.0, 'FG%': 0.0, '3P': 0.0, '3PA': 0.0, '3P%': 0.0,
    'FT': 0.0, 'FTA': 0.0, 'FT%': 0.0, 'ORB': 0.0, 'DRB': 0.0, 'TRB': 0.0, 'AST': 0.0,
    'STL': 0.0, 'BLK': 0.0, 'TOV': 0.0, 'PF': 0.0, 'PTS': 0.0
}
# Weight profiles for different focuses (offense, defense, balanced)
weight_profiles = {
    'offense': {
        'MP': 6.0, 'FG': 7.0, 'FGA': 5.0, 'FG%': 6.0, '3P': 9.0, '3PA': 5.0,
        '3P%': 8.0, 'FT': 4.0, 'FTA': 3.0, 'FT%': 7.0, 'ORB': 5.0, 'DRB': 2.0, 
        'TRB': 4.0, 'AST': 7.0, 'STL': 4.0, 'BLK': 4.0, 'TOV': 3.0, 'PF': 2.0,
        'PTS': 8.0
    },
    'defense': {
        'MP': 6.0, 'FG': 4.0, 'FGA': 3.0, 'FG%': 5.0, '3P': 4.0, '3PA': 3.0,
        '3P%': 4.0, 'FT': 4.0, 'FTA': 3.0, 'FT%': 4.0, 'ORB': 7.0, 'DRB': 8.0, 
        'TRB': 8.0, 'AST': 5.0, 'STL': 9.0, 'BLK': 9.0, 'TOV': 2.0, 'PF': 6.0,
        'PTS': 4.0
    },
    'balanced': {
        'MP': 7.0, 'FG': 6.0, 'FGA': 6.0, 'FG%': 6.0, '3P': 6.0, '3PA': 6.0, 
        '3P%': 7.0, 'FT': 6.0, 'FTA': 6.0, 'FT%': 6.0, 'ORB': 6.0, 'DRB': 6.0, 
        'TRB': 6.0, 'AST': 6.0, 'STL': 6.0, 'BLK': 6.0, 'TOV': 4.0, 'PF': 5.0, 
        'PTS': 6.0
    }
}
selected_profile = 'balanced'
raw_weights = weight_profiles[selected_profile]

# Normalize weights
total_weight = sum(raw_weights.values())
weights = {stat: value / total_weight for stat, value in raw_weights.items()}

# Scrape College Basketball Ref site
url = f'https://www.sports-reference.com/cbb/players/{college_player_id}.html'
response = requests.get(url)
html_content = response.content
soup = BeautifulSoup(html_content, 'html.parser')

# Find the relevant div and table
div = soup.find('div', id='div_players_per_game')
if div:
    table = div.find('table')
    if table:
        college_player_stats_df = pd.read_html(str(table))[0]
        
        if 'Player' in college_player_stats_df.columns:
            college_player_stats_df['Player'] = college_player_stats_df['Player'].apply(lambda x: unidecode(str(x)))
        
        # Rename columns
        college_player_stats_df = college_player_stats_df.rename(columns=column_mapping)

        # Check if 'Season' column exists
        if 'Season' in college_player_stats_df.columns:
            career_index = college_player_stats_df[college_player_stats_df['Season'] == 'Career'].index
            if not career_index.empty:
                latest_stats_index = career_index[0] - 1
                if latest_stats_index >= 0:
                    latest_stats = college_player_stats_df.iloc[latest_stats_index]
                    for stat in college_player.keys():
                        if stat in latest_stats.index:
                            college_player[stat] = latest_stats[stat]

                    college_player["MP"] *= 1.2
                    college_player["PTS"] *= 1.15
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

results = []
dir = './sample_DB/nba_clean_data'

# Convert college player stats to weighted NumPy array
college_stats = np.array([college_player.get(stat, 0) * weights.get(stat, 0) for stat in column_mapping.values()]).reshape(1, -1)

# Process each CSV file
for year in range(2015, 2025):
    csv_file = os.path.join(dir, f'{year}NBAPlayerStats_HprDNA.csv')
    if os.path.exists(csv_file):
        query = f"SELECT * FROM \"{csv_file}\""
        season_data = duckdb.query(query).df()
        stat_columns = list(column_mapping.values())
        
        if all(col in season_data.columns for col in stat_columns):
            nba_stats = season_data[stat_columns].fillna(0)
            college_player_pos = latest_stats.get('Position', 'Unknown')
            pos_mapping = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C']}
            nba_positions = pos_mapping.get(college_player_pos, [])

            if nba_positions:
                filtered_season_data = season_data[season_data['Position'].isin(nba_positions)].reset_index(drop=True)
                filtered_nba_stats = filtered_season_data[stat_columns].fillna(0)
                weighted_nba_stats = filtered_nba_stats.apply(lambda row: row * np.array([weights.get(stat, 0) for stat in stat_columns]), axis=1)
                distances = weighted_nba_stats.apply(lambda row: euclidean(row, college_stats.flatten()), axis=1)
                distances_df = pd.DataFrame({'Distance': distances.values}, index=filtered_season_data.index)
                min_dist_index = np.argmin(distances)

                if min_dist_index in distances_df.index:
                    distance_percentage = (1 / (1 + distances_df.loc[min_dist_index, 'Distance'])) * 100
                    most_similar_player_df = pd.DataFrame([filtered_season_data.iloc[min_dist_index]], columns=filtered_season_data.columns)
                    most_similar_player_df.loc[most_similar_player_df.index[0], 'Similarity (%)'] = f'{distance_percentage:.2f}%'
                    results.append(most_similar_player_df)
            else:
                print(f"No matching NBA positions found for college position: {college_player_pos}.")
        else:
            print(f"One or more columns from {stat_columns} are missing from {csv_file}.")
    else:
        print(f"File not found: {csv_file}")

# Combine results
most_similar_players = pd.concat(results, ignore_index=True)

if not most_similar_players.empty:
    sorted_most_similar_players = most_similar_players.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)
    print("\nNBA Most Similar Players By Year:")
    print(sorted_most_similar_players)
else:
    print("No matching players were found across the years provided.")
