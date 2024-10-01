import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

college_player_id = 'jasper-floyd-1'

selected_profile = 'offense'

csv_file_path = './sample_DB/college_data/college_basketball_players.csv'

def init_college_player_info(college_player_id, csv_file_path):

    # in - college player id, csv path
    # out - college player name, college player dict

    df = pd.read_csv(csv_file_path)

    row = df[df['playerId'] == college_player_id]

    if not row.empty:
        college_player_name = row['playerName'].values[0]

    college_player_dict = {
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

    return college_player_name, college_player_dict

def init_weight_profiles(selected_profile):

    # in - selected weight profile
    # out - corresponding weight profile dict

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

    raw_weights = weight_profiles[selected_profile]

    total_weight = sum(raw_weights.values())
    weights = {stat: value / total_weight for stat, value in raw_weights.items()}

    return weights

def scrape_college_stats(college_player_id, college_player_name, college_player_dict):

    #in - college id, college player name, college dict
    #out - college latest stats, college dict 

    url = f'https://www.sports-reference.com/cbb/players/{college_player_id}.html'
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')
    div = soup.find('div', id='div_players_per_game')

    if div:
        table = div.find('table')

        if table:
            college_player_stats_df = pd.read_html(str(table))[0]

            print(f"{college_player_name} | Stats:")
            print(college_player_stats_df)

            if 'Season' in college_player_stats_df.columns:
                career_index = college_player_stats_df[college_player_stats_df['Season'] == 'Career'].index

                if not career_index.empty:
                    college_latest_stats_index = career_index[0] - 1

                    if college_latest_stats_index >= 0:
                        college_latest_stats = college_player_stats_df.iloc[college_latest_stats_index]

                        for stat in college_player_dict.keys():
                            if stat in college_latest_stats.index:
                                college_player_dict[stat] = college_latest_stats[stat]

                        print(f"\n{college_player_name}'s Statline for Euclidean Distance Analysis:")
                        print(pd.DataFrame([college_latest_stats], columns=college_player_stats_df.columns))

                        # College player dict adjustments (NCAA => NBA)
                        college_player_dict["MP"] *= 1.17  # 40 vs 48 total min
                        college_player_dict["PTS"] *= 1.15  # skew scoring for better offensive player matches
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

    return college_latest_stats, college_player_dict

def find_nba_matches(college_latest_stats, weights, college_player_dict):

    #in - college latest stats, weights, college dict
    #out - nba matches

    college_stats = np.array([college_player_dict[stat] * weights[stat] for stat in college_player_dict.keys()]).reshape(1, -1)

    print("\nCustom Match Profile: " + str(selected_profile))

    print("\nCollege to NBA Conversion:")
    print("MP: " + str(round(college_player_dict["MP"], 2)) + " | PTS: " + str(round(college_player_dict["PTS"], 2)))

    results = []
    dir = './sample_DB/nba_raw_data'

    # 2015 to 2024

    for year in range(2015, 2025):
        csv_file = os.path.join(dir, f'{year}_NBAPlayerStats_HprDNA_raw.csv')
        
        if os.path.exists(csv_file):
            query = f"SELECT * FROM '{csv_file}'"
            season_data = duckdb.query(query).df()

            stat_columns = list(college_player_dict.keys())
            
            if all(col in season_data.columns for col in stat_columns):
                nba_stats = season_data[stat_columns].fillna(0)

                college_player_pos = college_latest_stats.get('Pos', 'UNKWN')

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

    nba_dna_matches = pd.concat(results, ignore_index=True)

    nba_dna_matches = nba_dna_matches.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

    return nba_dna_matches

def main():

    college_player_name, college_player_dict = init_college_player_info(college_player_id, csv_file_path)

    weights = init_weight_profiles(selected_profile)

    college_latest_stats, college_player_dict = scrape_college_stats(college_player_id, college_player_name, college_player_dict)

    nba_dna_matches = find_nba_matches(college_latest_stats, weights, college_player_dict)

    print(f"\n{college_player_name}'s NBA Player Matches (In Last Decade):")
    print(nba_dna_matches)

if __name__=="__main__":
    main()