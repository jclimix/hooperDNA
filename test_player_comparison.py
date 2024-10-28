import pandas as pd
import random
import numpy as np

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

def calculate_dna_match(college_player_df, nba_players_df):

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
    
    # Extract the college player's stats for the available columns
    college_stats = college_player_df[stat_columns].values.flatten().astype(float)

    dna_matches = []
    for _, nba_row in nba_filtered_df.iterrows():
        nba_stats = nba_row[stat_columns].values.flatten().astype(float)

        # Calculate Euclidean distance
        distance = np.linalg.norm(college_stats - nba_stats)
        
        # Normalize to a "DNA Match" percentage (similarity score)
        max_distance = np.sqrt(len(stat_columns)) * np.max([np.ptp(college_stats), np.ptp(nba_stats)])
        if max_distance == 0:
            max_distance = 1
        similarity_score = 100 * (1 - (distance / max_distance))

        similarity_score = round(similarity_score, 1)
        
        dna_matches.append(similarity_score)

    # Add DNA Match column to the filtered NBA DataFrame
    nba_filtered_df["DNA Match"] = dna_matches

    return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

college_player_stats_df = create_college_player()
print(college_player_stats_df)

nba_players_df = create_random_nba_players()
print(nba_players_df)

nba_with_dna_match = calculate_dna_match(college_player_stats_df, nba_players_df)
print(nba_with_dna_match)