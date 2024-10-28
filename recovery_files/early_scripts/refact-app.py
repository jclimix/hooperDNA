import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import re
import csv
import json
from dotenv import load_dotenv
import os, logging, boto3, pandas as pd, io
from loguru import logger
import random
import requests
import re
from bs4 import BeautifulSoup
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv('../secrets/s3-hooperdna/.env')

def read_csv_from_s3(bucket_name, key):

    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))

    obj = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df

def get_college_player_name(player_id):
    df = read_csv_from_s3('hooperdna-storage', 'college_data/college_basketball_players.csv')
    row = df[df["playerId"] == player_id]
    return row["playerName"].values[0] if not row.empty else None

def scrape_college_data(player_id):
    
    url = f'https://www.sports-reference.com/cbb/players/{player_id}.html'
    response = requests.get(url)
    
    # Return early if the request fails
    if response.status_code != 200:
        logger.error("Failed to retrieve the webpage. Status code: %s", response.status_code)
        return None, None, "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    
    # Step 1: Extract the player's height
    height_pattern = re.compile(r"([4-8]-\d{1,2})")
    height_element = soup.find("span", string=height_pattern)
    college_player_height = height_element.text.strip() if height_element else None
    if college_player_height:
        logger.info(f"College player height: {college_player_height}")
    else:
        logger.error("Height element not found.")
    
    # Step 2: Extract the player stats table (no changes made here)
    pattern = r'<div class="table_container tabbed current" id="div_players_per_game">(.*?)</div>'
    matches = re.search(pattern, content, re.DOTALL)

    if matches:
        div_content = matches.group(1)
        soup = BeautifulSoup(div_content, 'html.parser')
        table = soup.find('table')

        if table:
            df = pd.read_html(str(table))[0]  # [0] gets the first table
            df = df.iloc[[-2]]  # Extract the most recent season stats
        else:
            logger.error("Table not found within the extracted div content.")
            df = None
    else:
        logger.error("Div with id 'div_players_per_game' not found.")
        df = None
    
    soup = BeautifulSoup(content, 'html.parser')
    target_id = "meta"
    target_section = soup.find(id=target_id)
    college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"  # Default image link

    if target_section:
        media_items = target_section.find_all("div", class_="media-item")

        if media_items:
            for index, item in enumerate(media_items, 1):
                img_tag = item.find("img")
                if img_tag and "src" in img_tag.attrs:
                    college_image_link = img_tag["src"]
                    logger.info(f"College player headshot link found: {college_image_link}")
                    break
                else:
                    logger.error(f"Image {index}: No image found.")
        else:
            logger.error(f"No media items found in the section with id '{target_id}'.")
    else:
        logger.error(f"No section found with id '{target_id}'.")

    return college_player_height, df, college_image_link

def adjust_stats(df):

    columns_to_adjust = [
        'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
        'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
        'BLK', 'TOV', 'PF', 'PTS'
    ]
    
    # Create a copy of the DataFrame to avoid modifying the original
    adjusted_df = df.copy()
    
    # Multiply each specified column by 1.18 if it exists in the DataFrame
    for col in columns_to_adjust:
        if col in adjusted_df.columns:
            adjusted_df[col] = adjusted_df[col] * 1.2
            
    return adjusted_df

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


def load_nba_data(year):
    df = read_csv_from_s3('hooperdna-storage', f'nba_raw_data/{year}_NBAPlayerStats_HprDNA_raw.csv')
    return df

def find_matches_before_college_player(year):
        
    # Initialize all_nba_matches as an empty DataFrame
    all_nba_matches = pd.DataFrame()

    year = int(year[:4])
    last_n_years = 20

    if (year - last_n_years) < 1970:
        start_year = 1970
    else:
        start_year = year - last_n_years
    end_year = year

    for year in range(start_year, end_year):

        # Load NBA data for the specific year
        nba_players_df = load_nba_data(year=year)

        # Calculate DNA match
        nba_with_dna_match = calculate_dna_match(adjusted_college_stats_df, nba_players_df, weights_df)
        
        # Select the top match
        top_nba_match = nba_with_dna_match.iloc[[0]]

        # Add any missing columns from top_nba_match to all_nba_matches
        for col in top_nba_match.columns:
            if col not in all_nba_matches.columns:
                all_nba_matches[col] = 0

        # Append the top match to all_nba_matches
        all_nba_matches = pd.concat([all_nba_matches, top_nba_match], ignore_index=True)
    
    all_nba_matches = all_nba_matches.sort_values(by="DNA Match", ascending=False)

    return all_nba_matches


def scrape_nba_player_data(nba_match_player_name):

    # Read player ID data from S3
    df = read_csv_from_s3('hooperdna-storage', 'nba_player_data/nba_players_n_ids.csv')
    
    # Find the player ID based on the name
    player_row = df[df["playerName"].str.lower() == nba_match_player_name.lower()]
    if player_row.empty:
        logger.error(f"No player found with name '{nba_match_player_name}'.")
        return None, None

    nba_player_id = player_row["playerId"].values[0]
    first_char_nba_id = nba_player_id[0]

    # Construct the player's URL
    url = f"https://www.basketball-reference.com/players/{first_char_nba_id}/{nba_player_id}.html"

    # Default values for image link and height
    img_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"
    height = None

    try:
        # Request and parse the HTML content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the 'meta' section for both the image link and height
        metadata = soup.find(id="meta")

        if metadata:
            # Extract image link if available
            media_items = metadata.find_all("div", class_="media-item")
            for item in media_items:
                img_tag = item.find("img")
                if img_tag and "src" in img_tag.attrs:
                    img_link = img_tag["src"]
                    break

            # Extract height from metadata text
            height_pattern = re.compile(r'([4-8]-\d{1,2})')
            height_match = height_pattern.search(metadata.text)
            if height_match:
                height = height_match.group(0)
            else:
                logger.warning("Height pattern not found in metadata.")
        else:
            logger.error(f"No metadata section found on the page for player ID '{nba_player_id}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error while retrieving the webpage: {e}")
        return None, None

    return img_link, height

college_player_id = 'zach-edey-1'
selected_profile = 'offense'

college_player_name = get_college_player_name(college_player_id)
print(f"College Player Name: {college_player_name}")

college_player_height, college_player_stats_df, college_headshot_link = scrape_college_data(college_player_id)
print(college_player_stats_df)

adjusted_college_stats_df = adjust_stats(college_player_stats_df)
print(adjusted_college_stats_df)

college_player_season = str(college_player_stats_df['Season'].values[0])
college_player_position = str(college_player_stats_df['Pos'].values[0])
print(f"College Player Season: {college_player_season}")
print(f"College Player Position: {college_player_position}")

weights_df = create_weights_df(selected_profile)
#print(weights_df)

all_nba_matches = find_matches_before_college_player(college_player_season)
top_10_nba_matches = all_nba_matches.head(10)

with pd.option_context('display.max_columns', None):
    print(top_10_nba_matches)

top_1_nba_match_name = top_10_nba_matches["Player"].iloc[0]
top_1_nba_match_season = top_10_nba_matches["Season"].iloc[0]
top_1_nba_match_pos = top_10_nba_matches["Pos"].iloc[0]
top_1_nba_match_stats = top_10_nba_matches.iloc[[0]]

print(f"Top NBA Match Name: {top_1_nba_match_name}")
print(f"Top NBA Match Season: {top_1_nba_match_season}")
print(f"Top NBA Match Position: {top_1_nba_match_pos}")

top_1_nba_match_headshot_link, top_1_nba_match_height = scrape_nba_player_data(top_1_nba_match_name)

print(f"Top NBA Match Height: {top_1_nba_match_height}")
print(f"Top NBA Match Headshot Link: {top_1_nba_match_headshot_link}")

college_nba_join_stats = pd.concat([college_player_stats_df, top_1_nba_match_stats], axis=0, join='outer', ignore_index=True)
columns_to_remove = ['Conf', 'Class', 'Rk', 'Player', 'Age', 'DNA Match']
college_nba_join_stats = college_nba_join_stats.drop(columns=columns_to_remove)

print(top_1_nba_match_stats)
