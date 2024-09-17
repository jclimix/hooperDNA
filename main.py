import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

college_player_id = 'jasper-floyd-1'

csv_file_path = './sample_DB/college_data/college_basketball_players.csv'

df = pd.read_csv(csv_file_path)

row = df[df['playerId'] == college_player_id]

if not row.empty:
    player_name = row['playerName'].values[0]

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

# Select the weight profile to use (e.g., 'offense-heavy', 'defense-heavy', or 'balanced')
selected_profile = 'offense'
raw_weights = weight_profiles[selected_profile]

# Normalize the selected weights so they sum to 100% (1.0)
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

        print(f"{player_name} | Stats:")
        print(college_player_stats_df)

        # if the 'Season' column exists
        if 'Season' in college_player_stats_df.columns:
            career_index = college_player_stats_df[college_player_stats_df['Season'] == 'Career'].index

            if not career_index.empty:
                latest_stats_index = career_index[0] - 1

                if latest_stats_index >= 0:
                    latest_stats = college_player_stats_df.iloc[latest_stats_index]

                    for stat in college_player.keys():
                        if stat in latest_stats.index:
                            college_player[stat] = latest_stats[stat]

                    print(f"\n{player_name}'s Statline for Euclidean Distance Analysis:")
                    print(pd.DataFrame([latest_stats], columns=college_player_stats_df.columns))

                    # College stat adjustments (NCAA => NBA)
                    college_player["MP"] *= 1.17  # 40 vs 48 total min
                    college_player["PTS"] *= 1.15  # skew scoring for better offensive player matches
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

print("\nCustom Match Profile: " + str(selected_profile))

print("\nCollege to NBA Conversion:")
print("MP: " + str(round(college_player["MP"], 2)) + " | PTS: " + str(round(college_player["PTS"], 2)))

results = []
dir = './sample_DB/nba_raw_data'

# 2015 to 2024
for year in range(2015, 2025):
    csv_file = os.path.join(dir, f'{year}_NBAPlayerStats_HprDNA_raw.csv')
    
    if os.path.exists(csv_file):
        query = f"SELECT * FROM '{csv_file}'"
        season_data = duckdb.query(query).df()

        stat_columns = list(college_player.keys())
        
        if all(col in season_data.columns for col in stat_columns):
            nba_stats = season_data[stat_columns].fillna(0)

            college_player_pos = latest_stats.get('Pos', 'Unknown')

            pos_mapping = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C']}
            nba_positions = pos_mapping.get(college_player_pos, [])

            if nba_positions:
                filtered_season_data = season_data[season_data['Pos'].isin(nba_positions)].reset_index(drop=True)
                filtered_nba_stats = filtered_season_data[stat_columns].fillna(0)

                weighted_nba_stats = filtered_nba_stats.apply(lambda row: row * np.array([weights[stat] for stat in stat_columns]), axis=1)

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
            print(f"One or more columns from {stat_columns} are missing in {csv_file}.")
    else:
        print(f"File not found: {csv_file}")

# Combine all result DataFrames into a single DataFrame
nba_dna_matches = pd.concat(results, ignore_index=True)

# Sort by similarity score (descending)
nba_dna_matches = nba_dna_matches.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

print(f"\n{player_name}'s NBA Player Matches (In Last Decade):")
print(nba_dna_matches)