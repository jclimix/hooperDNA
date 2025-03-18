import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
from sql_utils.sql_transfers import *

def scrape_player(player_id):
    """Scrapes the table with id 'players_per_game' from the given URL and converts it into a DataFrame."""

    url = f"https://www.sports-reference.com/cbb/players/{player_id}.html"

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')

    return soup

def scrape_stats(soup):
    table = soup.find('table', {'id': 'players_per_game'})
    
    if table is None:
        raise ValueError("Table with id 'players_per_game' not found on the page.")
    
    df = pd.read_html(StringIO(str(table)))[0]

    last_season_df = df.iloc[[i - 1 for i in df.index if df.loc[i, "Season"] == "Career"]] if "Career" in df["Season"].values else pd.DataFrame()


    return last_season_df

def scrape_player_height(soup):

    # find the paragraph containing height
    for p in soup.find_all("p"):
        if p.find("span") and "-" in p.get_text():
            height = p.find("span").text.strip()
            return height

    return None

def get_college_player_name(df, college_player_id):
    """
    Finds the player name in a DataFrame based on player_id.

    :param df: Pandas DataFrame containing player data.
    :param player_id: The player ID to search for.
    :return: Player name if found, otherwise None.
    """
    row = df.loc[df["player_id"] == college_player_id, "player_name"]
    
    if not row.empty:
        return row.iloc[0]  # return the first match
    
    return None

def season_to_year(season):
    """
    Converts an NBA season string (e.g., '2020-21', '1999-00') to the ending year (e.g., 2021, 2000).
    
    :param season: NBA season string (format: 'YYYY-YY').
    :return: Integer representing the ending year of the season.
    """
    try:
        start_year, end_year = map(int, season.split("-"))
        return 2000 if end_year == 0 else (start_year // 100) * 100 + end_year
    except (ValueError, AttributeError, IndexError):
        return None

def year_to_season(year):
    """
    Converts a year integer (e.g., 2000) to an NBA season format (e.g., '1999-00').

    :param year: Integer representing the ending year of the NBA season.
    :return: NBA season string in 'YYYY-YY' format.
    """
    try:
        start_year = year - 1
        end_year = str(year)[-2:]  # get last two digits of the year
        return f"{start_year}-{end_year}"
    except (ValueError, TypeError):
        return None 

def get_last_10_seasons(n):
    """
    Creates a list of integers with a growing distance between 1 and 10 from the given number n,
    excluding values less than 1970 (but including 1970).

    :param n: The starting integer.
    :return: A list of integers with increasing gaps up to 10, filtered for values >= 1970.
    """
    return [x for x in (n - i for i in range(1, 11)) if x >= 1970]

def filter_by_position(df, position_str):
    """
    Filters a DataFrame by keeping only rows where the 'position' column contains the given string.

    :param df: Pandas DataFrame with a 'position' column.
    :param position_str: String to filter by.
    :return: Filtered DataFrame.
    """
    return df[df["position"].str.contains(position_str, case=False, na=False)]

def consolidate_traded_players(df):
    """
    Consolidates players who played for multiple teams in a season by keeping only the row with the 
    most games played and replacing the 'NTM' entry with the latest team played for.

    :param df: Pandas DataFrame with 'player_id', 'team', and 'G' (games played) columns.
    :return: Modified DataFrame with consolidated player entries.
    """
    # find rows where 'team' contains 'TM' (2TM, 3TM, etc.)
    mask = df["team"].str.contains("TM", na=False)

    # dict to store the latest team for each traded player
    latest_team_map = {}

    # process each unique player with 'NTM' entries
    for player_id in df.loc[mask, "player_id"].unique():
        player_rows = df[df["player_id"] == player_id]

        # get the latest team (first occurrence in the DataFrame, assuming it's sorted with latest first)
        latest_team = player_rows[player_rows["team"] != "2TM"].iloc[0]["team"]

        # store latest team mapping
        latest_team_map[player_id] = latest_team

    # find the row where the player has the most games played (including 'NTM' row)
    idx_max_games = df.groupby("player_id")["games_played"].idxmax()

    # keep only the row where the player played the most games
    df_filtered = df.loc[idx_max_games].reset_index(drop=True)

    # replace the team with the latest team for players who had 'NTM'
    df_filtered.loc[df_filtered["player_id"].isin(latest_team_map.keys()), "team"] = df_filtered["player_id"].map(latest_team_map)

    return df_filtered

def rename_columns(df):
    """
    Renames the columns in the given DataFrame to be lowercase and written out for clarity.
    
    :param df: Pandas DataFrame with columns to rename.
    :return: DataFrame with renamed columns.
    """
    column_mapping = {
        "G": "games_played",
        "GS": "games_started",
        "MP": "minutes_played",
        "FG": "field_goals",
        "FGA": "field_goal_attempts",
        "FG%": "field_goal_pct",
        "3P": "three_pointers_made",
        "3PA": "three_point_attempts",
        "3P%": "three_point_pct",
        "2P": "two_pointers_made",
        "2PA": "two_point_attempts",
        "2P%": "two_point_pct",
        "eFG%": "effective_field_goal_pct",
        "FT": "free_throws_made",
        "FTA": "free_throw_attempts",
        "FT%": "free_throw_pct",
        "ORB": "offensive_rebounds",
        "DRB": "defensive_rebounds",
        "TRB": "total_rebounds",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "TOV": "turnovers",
        "PF": "personal_fouls",
        "PTS": "points",
        "Awards": "awards"
    }
    
    df = df.rename(columns=column_mapping)
    return df

def simple_calculate_dna_match(college_player_df, nba_players_df, weights_df):
    try:
        if len(college_player_df) != 1:
            raise ValueError("college_player_df should contain only one row.")
        
        logger.debug("Starting DNA match calculation...")
        
        # use NBA DataFrame as is
        nba_filtered_df = nba_players_df.copy()
        
        logger.debug(f"Processing NBA DataFrame with {len(nba_filtered_df)} rows.")

        # get stat columns that are common to both dfs
        stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
        
        # make sure to only use stat columns that exist in the weights df
        valid_stat_columns = [col for col in stat_columns if col in weights_df.index]
        
        if not valid_stat_columns:
            logger.error("No valid stat columns found that exist in all three DataFrames.")
            raise ValueError("No valid stat columns found that exist in all three DataFrames.")
            
        logger.debug(f"Using stat columns: {valid_stat_columns}")

        filtered_weights = weights_df.loc[valid_stat_columns].values.flatten()

        # normalize weights to prevent scale issues
        if np.linalg.norm(filtered_weights) != 0:
            filtered_weights /= np.linalg.norm(filtered_weights)

        # create a copy of college player data to modify
        college_player_projected = college_player_df.copy()
        
        # columns to increase by 1.2 to project development
        projection_columns = [
            'points', 'assists', 'blocks', 'steals', 'total_rebounds', 
            'defensive_rebounds', 'offensive_rebounds', 'minutes_played', 
            'field_goals_made', 'field_goals_attempted', 'two_pointers_made', 
            'two_pointers_attempted', 'three_pointers_made', 'three_pointers_attempted'
        ]
        
        # 1.2x increase to available projection columns
        for col in projection_columns:
            if col in college_player_projected.columns:
                college_player_projected[col] = college_player_projected[col] * 1.2
        
        # convert to numeric safely, coercing errors to NaN
        college_stats = pd.to_numeric(college_player_projected[valid_stat_columns].values.flatten(), errors='coerce')

        dna_matches = []
        for idx, nba_row in nba_filtered_df.iterrows():
            nba_stats = pd.to_numeric(nba_row[valid_stat_columns].values.flatten(), errors='coerce')

            valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)
            
            if not valid_indices.any():
                logger.debug(f"Skipping row {idx} due to no valid indices.")
                dna_matches.append(np.nan)  # add a values to maintain length
                continue
            
            college_stats_valid = college_stats[valid_indices]
            nba_stats_valid = nba_stats[valid_indices]
            weights_valid = filtered_weights[valid_indices]

            # apply weights
            weighted_college_stats = college_stats_valid * weights_valid
            weighted_nba_stats = nba_stats_valid * weights_valid

            # cosine similarity to find matches
            dot_product = np.dot(weighted_college_stats, weighted_nba_stats)
            norm_college = np.linalg.norm(weighted_college_stats)
            norm_nba = np.linalg.norm(weighted_nba_stats)

            if norm_college == 0 or norm_nba == 0:
                logger.debug(f"Skipping row {idx} due to zero norm.")
                dna_matches.append(np.nan)  # a value is added to maintain length
                continue

            cosine_similarity = dot_product / (norm_college * norm_nba)

            # apply a penalty for large stat differences
            absolute_differences = np.abs(weighted_college_stats - weighted_nba_stats)
            if np.max(absolute_differences) == 0:
                penalty_factor = 0
            else:
                penalty_factor = np.mean(absolute_differences) / np.max(absolute_differences)
            
            penalty_factor = min(1, penalty_factor)  # penalty is at most 1

            # adjust similarity score with penalty
            adjusted_similarity_score = cosine_similarity * (1 - penalty_factor)

            # convert to a similarity score between 0 and 100
            similarity_score = round(adjusted_similarity_score * 100, 1)

            # ensure similarity score is within 0-100
            similarity_score = max(0, min(100, similarity_score))
            
            dna_matches.append(similarity_score)

        if len(dna_matches) != len(nba_filtered_df):
            logger.error("Mismatch in length between DNA matches and NBA DataFrame.")
            raise ValueError("Mismatch in length between DNA matches and NBA DataFrame.")

        nba_filtered_df["DNA Match"] = dna_matches

        logger.debug("DNA match calculation completed successfully.")
        return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.error(f"An error occurred in calculate_dna_match: {e}")
        raise

def create_weights_df(profile):
    """Creates weight DataFrame for a profile ('offense', 'defense', or 'balanced')."""
    weight_profiles = {
        "offense": {
            "minutes_played": 2.0, "field_goals": 7.5, "field_goal_attempts": 6.0, "field_goal_pct": 8.0,
            "three_pointers_made": 7.0, "three_point_attempts": 5.5, "three_point_pct": 8.0,
            "two_pointers_made": 6.5, "two_point_attempts": 5.5, "two_point_pct": 7.0, 
            "effective_field_goal_pct": 8.5, "free_throws_made": 6.5, "free_throw_attempts": 5.5, 
            "free_throw_pct": 7.0, "offensive_rebounds": 4.5, "defensive_rebounds": 2.0, 
            "total_rebounds": 3.0, "assists": 7.5, "steals": 1.5, "blocks": 1.0, 
            "turnovers": 4.5, "personal_fouls": 1.5, "points": 9.5
        },
        "defense": {
            "minutes_played": 2.0, "field_goals": 2.0, "field_goal_attempts": 1.5, "field_goal_pct": 2.5,
            "three_pointers_made": 1.5, "three_point_attempts": 1.0, "three_point_pct": 2.0,
            "two_pointers_made": 1.5, "two_point_attempts": 1.0, "two_point_pct": 2.0, 
            "effective_field_goal_pct": 2.5, "free_throws_made": 1.0, "free_throw_attempts": 0.5, 
            "free_throw_pct": 1.5, "offensive_rebounds": 7.0, "defensive_rebounds": 8.5, 
            "total_rebounds": 8.0, "assists": 2.0, "steals": 9.0, "blocks": 9.5, 
            "turnovers": 6.0, "personal_fouls": 4.5, "points": 2.0
        },
        "balanced": {
            "minutes_played": 4.0, "field_goals": 5.0, "field_goal_attempts": 4.0, "field_goal_pct": 5.5,
            "three_pointers_made": 5.0, "three_point_attempts": 4.0, "three_point_pct": 5.5,
            "two_pointers_made": 5.0, "two_point_attempts": 4.0, "two_point_pct": 5.0, 
            "effective_field_goal_pct": 5.5, "free_throws_made": 4.5, "free_throw_attempts": 4.0, 
            "free_throw_pct": 5.0, "offensive_rebounds": 5.0, "defensive_rebounds": 5.5, 
            "total_rebounds": 6.0, "assists": 6.0, "steals": 5.5, "blocks": 5.5, 
            "turnovers": 5.0, "personal_fouls": 4.0, "points": 6.5
        }
    }
    
    return pd.DataFrame(list(weight_profiles[profile].items()), columns=["stat", "weight"]).set_index("stat")

def remove_duplicate_players(df):
    return df.drop_duplicates(subset="player_id", keep="first")

def filter_games_played(df):
    df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce")
    return df[df["games_played"] >= 50]

def process_dna_match():
    college_player_ids_df = extract_table_to_df('college_player_ids', 'college')

    soup = scrape_player(college_player_id)
    college_player_name = get_college_player_name(college_player_ids_df, college_player_id)
    college_player_height = scrape_player_height(soup)
    college_player_stats_df = scrape_stats(soup)

    latest_season = college_player_stats_df["Season"].iloc[0]

    college_position = college_player_stats_df["Pos"].iloc[0]

    latest_year = season_to_year(latest_season)

    last_10_list = get_last_10_seasons(latest_year)

    nba_stats_dict = {year: extract_table_to_df(f'{year}_reg_season_stats', 'per_game_stats') for year in last_10_list}

    for year in last_10_list:
        nba_stats_dict[year] = consolidate_traded_players(nba_stats_dict[year])
        nba_stats_dict[year] = filter_by_position(nba_stats_dict[year], college_position)

    college_player_stats_df = rename_columns(college_player_stats_df)

    logger.info(f"College Player Name: {college_player_name}")
    logger.info(f"College Player Height: {college_player_height}")
    logger.info(f"College Player Position: {college_position}")
    logger.info(f"Latest Year: {latest_year}")
    logger.info(f"Latest Season: {latest_season}")
    logger.info(f"Last 10 Years: {last_10_list}")
    logger.info(f"College Player Stats:\n{college_player_stats_df.head()}")

    df_weights_offense = create_weights_df(algo_weight)
    final_matches_df = pd.DataFrame()

    for year in last_10_list:
        matches_df = simple_calculate_dna_match(college_player_stats_df, nba_stats_dict[year], df_weights_offense)
        season = year_to_season(year)
        matches_df.insert(1, "season", [season for _ in range(len(matches_df))])
        matches_df = matches_df[:10]
        final_matches_df = pd.concat([final_matches_df, matches_df], ignore_index=True)

    final_matches_df = final_matches_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

    final_matches_df = remove_duplicate_players(final_matches_df)
    final_matches_df = filter_games_played(final_matches_df)
    top_3_nba_matches_df = final_matches_df[:3]

    logger.info(f"Top 3 NBA Matches:\n{top_3_nba_matches_df.head()}")

if __name__ == '__main__':

    college_player_id = 'caitlin-clark-1'
    algo_weight = 'offense'

    process_dna_match()