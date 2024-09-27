import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import re

def move_column(df, column_name, new_position):
    column = df.pop(column_name)
    
    df.insert(new_position, column_name, column)
    
    return df

def remove_column(df, column_name):
    df = df.drop(columns=[column_name])
    
    return df

def extract_first_row(df):
    new_df = df.iloc[[0]].reset_index(drop=True)
    
    return new_df

def get_player_id(player_name, csv_file):
    df = pd.read_csv(csv_file)

    matching_players = df[df['playerName'].str.lower() == player_name.lower()]

    if not matching_players.empty:
        player_id = matching_players.iloc[-1]['playerId']  
        return player_id
    else:
        return None 

def scrape_nba_player_data(url):
    target_id = 'meta' 
    img_link = None
    height = None

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        target_section = soup.find(id=target_id)

        if not target_section:
            print(f"No section found with id '{target_id}'.")
        else:
            media_items = target_section.find_all('div', class_='media-item')

            if media_items:
                for index, item in enumerate(media_items, 1):
                    img_tag = item.find('img')
                    if img_tag and 'src' in img_tag.attrs:
                        img_link = img_tag['src']
                        break
                    else:
                        print(f"Image {index}: No image found.")
            else:
                print(f"No media items found in the section with id '{target_id}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error while scraping the webpage: {e}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        height_pattern = re.compile(r'([4-8]-\d{1,2})')

        height_element = soup.find('p', string=height_pattern)

        if height_element:

            height_match = height_pattern.search(height_element.text)
            if height_match:
                height = height_match.group(0)
            else:
                print("Height not found.")
        else:
            print("Height element not found.")

    except requests.exceptions.RequestException as e:
        print(f"Error while scraping the webpage: {e}")

    return img_link, height

selected_college_player = 'Zach Edey'

csv_file_path = './sample_DB/college_data/college_basketball_players.csv'

college_player_id = get_player_id(selected_college_player, csv_file_path)

df = pd.read_csv(csv_file_path)

row = df[df['playerId'] == college_player_id]

if not row.empty:
    college_player_name = row['playerName'].values[0]

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

selected_profile = 'offense'
raw_weights = weight_profiles[selected_profile]

total_weight = sum(raw_weights.values())
weights = {stat: value / total_weight for stat, value in raw_weights.items()}

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
                latest_stats_index = career_index[0] - 1

                if latest_stats_index >= 0:
                    latest_stats = college_player_stats_df.iloc[latest_stats_index]

                    for stat in college_player.keys():
                        if stat in latest_stats.index:
                            college_player[stat] = latest_stats[stat]

                    print(f"\n{college_player_name}'s Statline for Euclidean Distance Analysis:")
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

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.content, 'html.parser')

# Regular expression pattern to match heights (ranging from 4-0 to 8-11)
height_pattern = re.compile(r'([4-8]-\d{1,2})')

height_element = soup.find('span', string=height_pattern)

if height_element:
    college_player_height = height_element.text.strip()
    # print(f"College player height: {college_player_height}")
else:
    print("Height element not found.")

# pull college player headshot

target_id = 'meta'
college_image_link = 'https://i.ibb.co/vqkzb0m/temp-player-pic.png'

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.content, 'html.parser')

target_section = soup.find(id=target_id)

if not target_section:
    print(f"No section found with id '{target_id}'.")

media_items = target_section.find_all('div', class_='media-item')

if media_items:
    for index, item in enumerate(media_items, 1):
        img_tag = item.find('img')
        if img_tag and 'src' in img_tag.attrs:
            college_image_link = img_tag['src']
            # print(f"Image {index} link: {college_image_link}")
        else:
            print(f"Image {index}: No image found.")
else:
    print(f"No media items found in the section with id '{target_id}'.")

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

            college_player_position = latest_stats.get('Pos', 'Unknown')

            pos_mapping = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C']}
            nba_positions = pos_mapping.get(college_player_position, [])

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
                print(f"No matching NBA positions found for college position: {college_player_position}.")
        else:
            print(f"One or more columns from {stat_columns} are missing in {csv_file}.")
    else:
        print(f"File not found: {csv_file}")

nba_dna_matches = pd.concat(results, ignore_index=True)

nba_dna_matches = nba_dna_matches.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)

nba_dna_matches = move_column(nba_dna_matches, 'Similarity (%)', 1)

nba_dna_matches = move_column(nba_dna_matches, 'Player', 0)

nba_dna_matches = remove_column(nba_dna_matches, 'Rk')
nba_dna_matches = remove_column(nba_dna_matches, 'G')
nba_dna_matches = remove_column(nba_dna_matches, 'GS')

nba_dna_matches = move_column(nba_dna_matches, 'PTS', 7)

first_nba_match = extract_first_row(nba_dna_matches)

print(f"\nBest NBA Player Matches")
print(first_nba_match)

print(f"\n{college_player_name}'s NBA Player Matches (In Last Decade):")
print(nba_dna_matches)

nba_match_player_name = first_nba_match['Player'].iloc[0]

def find_nba_player_id(player_name, csv_file):
    csv_file = './sample_DB/nba_player_data/nba_players_n_ids.csv'
    df = pd.read_csv(csv_file)
    
    player_row = df[df['playerName'].str.lower() == player_name.lower()]
    player_id = player_row['playerId'].values[0]
    return player_id

nba_player_id = find_nba_player_id(nba_match_player_name, csv_file)
first_char_nba_id = nba_player_id[0]


nba_url = f'https://www.basketball-reference.com/players/{first_char_nba_id}/{nba_player_id}.html' 

nba_image_link = 'https://i.ibb.co/vqkzb0m/temp-player-pic.png'
nba_image_link, nba_player_height = scrape_nba_player_data(nba_url)

nba_player_position = first_nba_match['Pos'].iloc[0]
dna_match_percentage = first_nba_match['Similarity (%)'].iloc[0]

print("NBA Player Match Name: " + nba_match_player_name)
print("NBA Player Link: " + str(nba_image_link))
print("NBA Player Height: " + (nba_player_height))
print("NBA Player Position: " + (nba_player_position))
print("DNA Match: " + (dna_match_percentage))
print("College Player Name: " + college_player_name)
print("College Player Height: " + (college_player_height))
print("College Player Position: " + (college_player_position))
print("College Player Link: " + str(college_image_link))