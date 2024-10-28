import pandas as pd
import random
import numpy as np
import requests
import re
from bs4 import BeautifulSoup

def scrape_college_data(id):
        
    # Step 1: Download the webpage and store the content in memory
    url = f'https://www.sports-reference.com/cbb/players/{id}.html'  # Replace with the actual URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Store the content directly in memory
        content = response.text
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)
        exit()

    # Step 2: Use regex to find the JavaScript block containing the target div
    # The regex pattern searches for the div with id 'div_players_per_game' and captures the HTML content within it
    pattern = r'<div class="table_container tabbed current" id="div_players_per_game">(.*?)</div>'
    matches = re.search(pattern, content, re.DOTALL)

    # Check if the pattern was found
    if matches:
        # Extract the div's content (which includes the table HTML)
        div_content = matches.group(1)
        
        # Step 3: Parse the extracted HTML snippet to find the table
        soup = BeautifulSoup(div_content, 'html.parser')
        table = soup.find('table')
        
        if table:
            # Step 4: Convert the HTML table to a DataFrame
            df = pd.read_html(str(table))[0]  # [0] gets the first table

            df = df.iloc[[-2]]
            
            #print(df)
            return df
        
        else:
            print("Table not found within the extracted div content.")
    else:
        print("Div with id 'div_players_per_game' not found in the JavaScript.")

def create_college_player():
    # Define the data for a sample player
    data = {
        "Season": ["2021-22", "2022-23", "2023-24"],
        "Team": ["Duke", "Duke", "Duke"],
        "Conf": ["ACC", "ACC", "ACC"],
        "Class": ["Freshman", "Freshman", "Sophomore"],
        "Pos": ["G", "G", "G"],
        #"G": [25, 29, 30],
        #"GS": [15, 20, 28],
        #"MP": [25.3, 30.1, 32.4],
        "FG": [4.2, 5.9, 6.5],
        "FGA": [10.5, 12.3, 13.2],
        "FG%": [40.0, 48.0, 49.2],
        #"3P": [1.8, 2.0, 2.3],
        #"3PA": [4.0, 5.3, 5.8],
        #"3P%": [32.1, 37.5, 39.7],
        "2P": [2.4, 3.9, 4.2],
        "2PA": [6.5, 7.0, 7.4],
        "2P%": [51.5, 55.7, 56.8],
        #"eFG%": [48.0, 52.6, 54.9],
        "FT": [2.1, 3.0, 2.8],
        "FTA": [2.8, 3.7, 3.5],
        "FT%": [75.0, 81.1, 80.2],
        #"ORB": [0.9, 1.0, 1.2],
        #"DRB": [2.8, 3.3, 3.5],
        "TRB": [3.7, 4.3, 4.7],
        "AST": [3.1, 4.8, 5.2],
        "STL": [1.0, 1.5, 1.8],
        #"BLK": [0.3, 0.6, 0.5],
        #"TOV": [2.0, 2.5, 2.2],
        "PF": [1.8, 2.1, 2.3],
        "PTS": [13.4, 16.8, 17.8],
        "Awards": ["None", "Freshman of the Year", "All-ACC"]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort DataFrame by 'Season' column in ascending order to have older years at the top
    df = df.sort_values(by="Season", ascending=True).reset_index(drop=True)

    df = df.iloc[[-1]]
    
    return df

def create_random_nba_players(num_players=10):
    # Define potential values for categorical data
    teams = ["Lakers", "Warriors", "Celtics", "Bulls", "Heat", "Nets"]
    positions = ["PG", "SG", "SF", "PF", "C"]
    awards_list = ["All-Star", "MVP", "Defensive Player of the Year", "All-NBA First Team", "Most Improved Player", "None"]

    # Create empty list to store player data
    players_data = []

    for _ in range(num_players):
        # Randomly generate data for a player
        season = f"{random.randint(2018, 2023)}-{random.randint(19, 24)}"
        team = random.choice(teams)
        pos = random.choice(positions)
        g = random.randint(60, 82)  # Games played
        gs = g  # Assume all games played were starts for simplicity
        mp = round(random.uniform(20.0, 38.0), 1)  # Minutes per game
        fg = round(random.uniform(5.0, 12.0), 1)  # Field goals made
        fga = round(fg + random.uniform(5.0, 10.0), 1)  # Field goals attempted
        fg_percent = round((fg / fga) * 100, 1) if fga > 0 else 0  # FG%
        three_p = round(random.uniform(1.0, 3.5), 1)  # 3-pointers made
        three_pa = round(three_p + random.uniform(1.0, 4.0), 1)  # 3-pointers attempted
        three_p_percent = round((three_p / three_pa) * 100, 1) if three_pa > 0 else 0  # 3P%
        two_p = round(fg - three_p, 1)  # 2-pointers made
        two_pa = round(fga - three_pa, 1)  # 2-pointers attempted
        two_p_percent = round((two_p / two_pa) * 100, 1) if two_pa > 0 else 0  # 2P%
        efg_percent = round(((fg + 0.5 * three_p) / fga) * 100, 1) if fga > 0 else 0  # eFG%
        ft = round(random.uniform(2.0, 7.0), 1)  # Free throws made
        fta = round(ft + random.uniform(0.5, 2.0), 1)  # Free throws attempted
        ft_percent = round((ft / fta) * 100, 1) if fta > 0 else 0  # FT%
        orb = round(random.uniform(0.5, 2.5), 1)  # Offensive rebounds
        drb = round(random.uniform(2.0, 8.0), 1)  # Defensive rebounds
        trb = round(orb + drb, 1)  # Total rebounds
        ast = round(random.uniform(1.0, 10.0), 1)  # Assists
        stl = round(random.uniform(0.5, 2.5), 1)  # Steals
        blk = round(random.uniform(0.5, 2.0), 1)  # Blocks
        tov = round(random.uniform(1.0, 4.0), 1)  # Turnovers
        pf = round(random.uniform(1.0, 3.5), 1)  # Personal fouls
        pts = round((fg * 2) + (three_p * 3) + ft, 1)  # Points
        awards = random.choice(awards_list)  # Random award or none

        # Append data to the list
        players_data.append({
            "Season": season,
            "Team": team,
            "Pos": pos,
            "G": g,
            "GS": gs,
            "MP": mp,
            "FG": fg,
            "FGA": fga,
            "FG%": fg_percent,
            "3P": three_p,
            "3PA": three_pa,
            "3P%": three_p_percent,
            "2P": two_p,
            "2PA": two_pa,
            "2P%": two_p_percent,
            "eFG%": efg_percent,
            "FT": ft,
            "FTA": fta,
            "FT%": ft_percent,
            "ORB": orb,
            "DRB": drb,
            "TRB": trb,
            "AST": ast,
            "STL": stl,
            "BLK": blk,
            "TOV": tov,
            "PF": pf,
            "PTS": pts,
            "Awards": awards
        })

    # Create DataFrame
    df = pd.DataFrame(players_data)
    
    # Sort by season for clarity
    df = df.sort_values(by="Season").reset_index(drop=True)
    
    return df

def create_weights_df(profile):

    weight_profiles = {
        "offense": {
            "MP": 2.0, "FG": 6.0, "FGA": 5.0, "FG%": 6.5, "3P": 5.5, "3PA": 4.5, "3P%": 6.5,
            "2P": 5.0, "2PA": 4.5, "2P%": 5.5, "eFG%": 6.0, "FT": 5.0, "FTA": 4.5, "FT%": 5.5,
            "ORB": 3.0, "DRB": 2.5, "TRB": 3.0, "AST": 6.0, "STL": 2.0, "BLK": 2.0, 
            "TOV": 3.0, "PF": 2.0, "PTS": 8.0
        },
        "defense": {
            "MP": 2.0, "FG": 3.0, "FGA": 2.5, "FG%": 3.5, "3P": 2.5, "3PA": 2.0, "3P%": 3.0,
            "2P": 2.5, "2PA": 2.0, "2P%": 3.0, "eFG%": 3.5, "FT": 2.0, "FTA": 1.5, "FT%": 2.5,
            "ORB": 5.5, "DRB": 6.0, "TRB": 6.5, "AST": 3.0, "STL": 6.5, "BLK": 7.0, 
            "TOV": 5.0, "PF": 3.5, "PTS": 3.0
        },
        "balanced": {
            "MP": 3.0, "FG": 4.5, "FGA": 4.0, "FG%": 5.0, "3P": 4.0, "3PA": 3.5, "3P%": 5.0,
            "2P": 4.0, "2PA": 3.5, "2P%": 4.5, "eFG%": 5.0, "FT": 4.0, "FTA": 3.5, "FT%": 4.5,
            "ORB": 4.0, "DRB": 4.5, "TRB": 5.0, "AST": 4.5, "STL": 4.0, "BLK": 4.0, 
            "TOV": 3.5, "PF": 3.0, "PTS": 5.5
        }
    }

    selected_profile = profile

    weights_df = pd.DataFrame(list(weight_profiles[selected_profile].items()), columns=["Stat", "Weight"]).set_index("Stat")
    return weights_df

def calculate_dna_match(college_player_df, nba_players_df, weights_df):

    
    if len(college_player_df) != 1:
        raise ValueError("college_player_df should contain only one row.")

    # Define position mapping
    position_mapping = {
        "G": ["PG", "SG"],
        "F": ["SF", "PF"],
        "C": ["C"]
    }
    
    # Get the college player's position
    college_position = college_player_df.iloc[0]["Pos"]
    if college_position not in position_mapping:
        raise ValueError("Invalid college position")

    # Filter NBA players to only those with positions matching the college player's position
    valid_nba_positions = position_mapping[college_position]
    nba_filtered_df = nba_players_df[nba_players_df["Pos"].isin(valid_nba_positions)].copy()
    
    # Identify common stat columns and convert to list
    stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
    
    # Filter weights to include only the columns available in college_player_df
    filtered_weights = weights_df.loc[stat_columns].values.flatten()

    # Extract the college player's stats for the available columns
    college_stats = college_player_df[stat_columns].values.flatten().astype(float)

    dna_matches = []
    for _, nba_row in nba_filtered_df.iterrows():
        nba_stats = nba_row[stat_columns].values.flatten().astype(float)

        # Calculate weighted Euclidean distance
        weighted_diff = (college_stats - nba_stats) * filtered_weights
        distance = np.linalg.norm(weighted_diff)
        
        # Normalize to a "DNA Match" percentage (similarity score)
        max_distance = np.sqrt(len(stat_columns)) * np.max([np.ptp(college_stats * filtered_weights), np.ptp(nba_stats * filtered_weights)])
        if max_distance == 0:
            max_distance = 1
        similarity_score = 100 * (1 - (distance / max_distance))

        similarity_score = round(similarity_score, 1)
        
        dna_matches.append(similarity_score)

    # Add DNA Match column to the filtered NBA DataFrame
    nba_filtered_df["DNA Match"] = dna_matches

    return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

college_player_id = 'zach-edey-1'
selected_profile = 'defense'

college_player_stats_df = scrape_college_data(college_player_id)
print(college_player_stats_df)

nba_players_df = create_random_nba_players()
print(nba_players_df)

weights_df = create_weights_df(selected_profile)
print(weights_df)

nba_with_dna_match = calculate_dna_match(college_player_stats_df, nba_players_df, weights_df)
print(nba_with_dna_match)