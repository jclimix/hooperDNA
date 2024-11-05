import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
import os, logging, boto3, pandas as pd, io
from loguru import logger
from algorithms import run_algorithm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Strategy pattern method for swapping algos

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
    
    # return early if request fails
    if response.status_code != 200:
        logger.error("Failed to retrieve the webpage. Status code: %s", response.status_code)
        return None, None, "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    
    height_pattern = re.compile(r"([4-8]-\d{1,2})")
    height_element = soup.find("span", string=height_pattern)
    college_player_height = height_element.text.strip() if height_element else None
    if college_player_height:
        logger.info(f"College player height: {college_player_height}")
    else:
        logger.error("Height element not found.")
    
    pattern = r'<div class="table_container tabbed current" id="div_players_per_game">(.*?)</div>'
    matches = re.search(pattern, content, re.DOTALL)

    if matches:
        div_content = matches.group(1)
        soup = BeautifulSoup(div_content, 'html.parser')
        table = soup.find('table')

        if table:
            df = pd.read_html(str(table))[0] 
            df = df.iloc[[-2]]  # most recent season stats
        else:
            logger.error("Table not found within the extracted div content.")
            df = None
    else:
        logger.error("Div with id 'div_players_per_game' not found.")
        df = None
    
    soup = BeautifulSoup(content, 'html.parser')
    target_id = "meta"
    target_section = soup.find(id=target_id)
    college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"  # default headshot-image link

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
        
    for col in columns_to_adjust:
        if col in df.columns:
            df[col] = round(df[col] * 1.13, 3) # multiply by 1.13 to convert college stats to NBA (pro)
            
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

def load_nba_data(year):
    df = read_csv_from_s3('hooperdna-storage', f'nba_raw_data/{year}_NBAPlayerStats_HprDNA_raw.csv')
    return df

def find_matches_before_college_player(year, adjusted_college_stats_df, weights_df, selected_algo):
        
    all_nba_matches = pd.DataFrame()

    year = int(year[:4])
    last_n_years = 20

    if (year - last_n_years) < 1970:
        start_year = 1970
    else:
        start_year = year - last_n_years
    end_year = year

    for year in range(start_year, end_year):

        nba_players_df = load_nba_data(year=year)

        # run_algo function (algo based on selected algo)
        nba_with_dna_match = run_algorithm(selected_algo, adjusted_college_stats_df, nba_players_df, weights_df)

        top_nba_match = nba_with_dna_match.iloc[[0]]

        for col in top_nba_match.columns:
            if col not in all_nba_matches.columns:
                all_nba_matches[col] = 0

        all_nba_matches = pd.concat([all_nba_matches, top_nba_match], ignore_index=True)
    
    all_nba_matches = all_nba_matches.sort_values(by="DNA Match", ascending=False)

    return all_nba_matches

def scrape_nba_player_data(nba_match_player_name):

    df = read_csv_from_s3('hooperdna-storage', 'nba_player_data/nba_players_n_ids.csv')
    
    player_row = df[df["playerName"].str.lower() == nba_match_player_name.lower()]
    if player_row.empty:
        logger.error(f"No player found with name '{nba_match_player_name}'.")
        return None, None

    nba_player_id = player_row["playerId"].values[0]
    first_char_nba_id = nba_player_id[0]

    url = f"https://www.basketball-reference.com/players/{first_char_nba_id}/{nba_player_id}.html"

    image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png" # default values for image link and height
    height = 'NA'

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        metadata = soup.find(id="meta")

        if metadata:
            media_items = metadata.find_all("div", class_="media-item")
            for item in media_items:
                img_tag = item.find("img")
                if img_tag and "src" in img_tag.attrs:
                    image_link = img_tag["src"]
                    break

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

    return image_link, height
import pandas as pd

def round_stats(df):
    # percentage_columns = ["FG%", "3P%", "2P%", "eFG%", "FT%"]
    int_columns = ["G", "GS"]
    float_columns = ["MP", "FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA", 
                     "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

    for column in df.columns:
        if column in int_columns:
            df[column] = df[column].apply(lambda x: int(round(x)) if pd.notnull(x) else x)
        elif column in float_columns:
            df[column] = df[column].apply(lambda x: round(x, 1) if pd.notnull(x) else x)

    return df
