#utils.py

import pandas as pd
from scipy.spatial.distance import euclidean
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

def read_csv_from_s3(bucket_name, key):

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))

    obj = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df


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

def round_dict_values(input_dict):
    rounded_dict = {}
    for key, value in input_dict.items():

        if isinstance(value, (int, float)):
            rounded_dict[key] = round(value, 2)
        else:
            rounded_dict[key] = value
    return rounded_dict

def csv_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict()

def df_to_dict(df):

    return {col: df[col].values[0] for col in df.columns}

def shift_df_col(df, col, pos):
    column_to_move = df.pop(col)
    df.insert(pos, col, column_to_move)

    return df

def shift_dict_key(d, key, new_position):
    if key not in d:
        raise KeyError(f"Key '{key}' not found in dictionary.")

    key_value_pair = {key: d.pop(key)}

    items = list(d.items())

    items.insert(new_position, list(key_value_pair.items())[0])

    return dict(items)

def get_player_id(player_name, csv_file):
    df = pd.read_csv(csv_file)

    matching_players = df[df["playerName"].str.lower() == player_name.lower()]

    if not matching_players.empty:
        player_id = matching_players.iloc[-1]["playerId"]
        return player_id
    else:
        return None

def csv_to_nested_dict(csv_file, key_column):
    df = pd.read_csv(csv_file)
    nested_dict = {}
    for _, row in df.iterrows():
        key = row[key_column]
        nested_dict[key] = row.drop(key_column).to_dict()
    return nested_dict


def scrape_nba_player_data(nba_match_player_name):

    df = read_csv_from_s3('hooperdna-storage', 'nba_player_data/nba_players_n_ids.csv')

    player_row = df[df["playerName"].str.lower() == nba_match_player_name.lower()]
    nba_player_id = player_row["playerId"].values[0]
    first_char_nba_id = nba_player_id[0]

    url = f"https://www.basketball-reference.com/players/{first_char_nba_id}/{nba_player_id}.html"

    target_id = "meta"
    img_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"
    height = None

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        target_section = soup.find(id=target_id)

        if not target_section:
            logger.error(f"No section found with id '{target_id}'.")
        else:
            media_items = target_section.find_all("div", class_="media-item")

            if media_items:
                for index, item in enumerate(media_items, 1):
                    img_tag = item.find("img")
                    if img_tag and "src" in img_tag.attrs:
                        img_link = img_tag["src"]
                        break
                    else:
                        logger.warning(f"Image {index}: No image found.")
            else:
                logger.error(f"No media items found in the section with id '{target_id}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error while scraping the webpage: {e}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        height_pattern = re.compile(r"([4-8]-\d{1,2})")

        height_element = soup.find("p", string=height_pattern)

        if height_element:

            height_match = height_pattern.search(height_element.text)
            if height_match:
                height = height_match.group(0)
            else:
                logger.error("Height not found.")
        else:
            logger.error("Height element not found.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error while scraping the webpage: {e}")

    return img_link, height

def generate_json_from_csv():

    csv_file = read_csv_from_s3('hooperdna-storage', 'college_data/college_basketball_players.csv') 
    json_file = './static/players.json'

    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    if not os.path.exists(json_file):
        data = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({'name': row['playerName'], 'id': row['playerId']})
        
        with open(json_file, mode='w') as file:
            json.dump(data, file)

def scrape_college_player_data(college_player_id, college_player, weight_profiles, selected_profile):
    raw_weights = weight_profiles[selected_profile]
    total_weight = sum(raw_weights.values())
    weights = {stat: value / total_weight for stat, value in raw_weights.items()}

    url = f"https://www.sports-reference.com/cbb/players/{college_player_id}.html"

    # Nested function to scrape stats
    def scrape_college_stats():
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        div = soup.find("div", id="div_players_per_game")

        if div:
            table = div.find("table")
            if table:
                college_player_stats_df = pd.read_html(str(table))[0]

                logger.info(f"{college_player_id} | Stats:")
                logger.info(college_player_stats_df)

                if "Season" in college_player_stats_df.columns:
                    career_index = college_player_stats_df[
                        college_player_stats_df["Season"] == "Career"
                    ].index

                    if not career_index.empty:
                        latest_stats_index = career_index[0] - 1
                        if latest_stats_index >= 0:
                            college_latest_stats = college_player_stats_df.iloc[latest_stats_index]

                            for stat in college_player.keys():
                                if stat in college_latest_stats.index:
                                    college_player[stat] = college_latest_stats[stat]

                            logger.info(f"\nStatline for Euclidean Distance Analysis:")
                            logger.info(pd.DataFrame([college_latest_stats], columns=college_player_stats_df.columns))

                            return college_player, college_latest_stats, college_player_stats_df
                        else:
                            logger.error("No valid row found before 'Career' row.")
                    else:
                        logger.error("'Career' row not found in the stats table.")
                else:
                    logger.error("'Season' column not found in the stats table.")
            else:
                logger.error("Table not found on the page.")
        else:
            logger.error("Div with player stats not found on the page.")
        return None

    def scrape_college_height():
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # regular expression pattern to match heights (ranging from 4-0 to 8-11)
        height_pattern = re.compile(r"([4-8]-\d{1,2})")
        height_element = soup.find("span", string=height_pattern)

        if height_element:
            college_player_height = height_element.text.strip()
            logger.info(f"College player height: {college_player_height}")
            return college_player_height
        else:
            logger.error("Height element not found.")
            return None

    def scrape_college_headshot():
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        target_id = "meta"
        target_section = soup.find(id=target_id)
        college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

        if target_section:
            media_items = target_section.find_all("div", class_="media-item")

            if media_items:
                for index, item in enumerate(media_items, 1):
                    img_tag = item.find("img")
                    if img_tag and "src" in img_tag.attrs:
                        college_image_link = img_tag["src"]
                        break
                    else:
                        logger.error(f"Image {index}: No image found.")
            else:
                logger.error(f"No media items found in the section with id '{target_id}'.")
        else:
            logger.error(f"No section found with id '{target_id}'.")
        
        logger.info(f"College player headshot link: {college_image_link}")
        return college_image_link

    def adjust_college_stats_to_weights():
        college_stats = np.array(
            [float(college_player[stat]) * weights[stat] for stat in college_player.keys()]
        ).reshape(1, -1)
        logger.info(f"Custom Match Profile: {selected_profile}")
        return college_stats

    stats, college_latest_stats, college_player_stats_df = scrape_college_stats()
    height = scrape_college_height()
    headshot = scrape_college_headshot()
    weighted_stats = adjust_college_stats_to_weights()

    return {
        "stats": stats,
        "college_latest_stats": college_latest_stats,
        "college_player_stats_df": college_player_stats_df,
        "height": height,
        "headshot": headshot,
        "weighted_stats": weighted_stats
    }

def find_nba_matches(college_player, college_latest_stats, college_stats, weight_profiles, selected_profile, read_csv_from_s3):
    nba_player_analysis_results = []

    raw_weights = weight_profiles[selected_profile]
    total_weight = sum(raw_weights.values())
    weights = {stat: value / total_weight for stat, value in raw_weights.items()}

    # 2015 to 2024
    for year in range(2015, 2025):
        csv_file = read_csv_from_s3('hooperdna-storage', f'nba_raw_data/{year}_NBAPlayerStats_HprDNA_raw.csv')

        if not csv_file.empty:
            season_data = csv_file
            stat_columns = list(college_player.keys())

            if all(col in season_data.columns for col in stat_columns):
                college_player_position = college_latest_stats.get("Pos", "Unknown")
                pos_mapping = {"G": ["PG", "SG"], "F": ["SF", "PF"], "C": ["C"]}
                nba_positions = pos_mapping.get(college_player_position, [])

                if nba_positions:
                    filtered_season_data = season_data[season_data["Pos"].isin(nba_positions)].reset_index(drop=True)
                    filtered_nba_stats = filtered_season_data[stat_columns].fillna(0)

                    weighted_nba_stats = filtered_nba_stats.apply(
                        lambda row: row * np.array([weights[stat] for stat in stat_columns]), axis=1
                    )

                    distances = weighted_nba_stats.apply(
                        lambda row: euclidean(row, college_stats.flatten()), axis=1
                    )

                    distances_df = pd.DataFrame({"Distance": distances.values}, index=filtered_season_data.index)
                    min_dist_index = np.argmin(distances)

                    if min_dist_index in distances_df.index:
                        distance_percentage = (1 / (1 + distances_df.loc[min_dist_index, "Distance"])) * 100

                        most_similar_player_df = pd.DataFrame(
                            [filtered_season_data.iloc[min_dist_index]],
                            columns=filtered_season_data.columns,
                        )

                        most_similar_player_df.loc[most_similar_player_df.index[0], "Similarity (%)"] = f"{distance_percentage:.2f}%"

                        nba_player_analysis_results.append(most_similar_player_df)
                else:
                    logger.error(f"No matching NBA positions found for college position: {college_player_position}.")
            else:
                logger.error(f"One or more columns from {stat_columns} are missing in the CSV file.")
        else:
            logger.error(f"File for year {year} not found.")

    if nba_player_analysis_results:
        nba_dna_matches = pd.concat(nba_player_analysis_results, ignore_index=True)

        nba_dna_matches = nba_dna_matches.sort_values(by="Similarity (%)", ascending=False).reset_index(drop=True)

        nba_dna_matches = move_column(nba_dna_matches, "Similarity (%)", 1)
        nba_dna_matches = move_column(nba_dna_matches, "Player", 0)
        nba_dna_matches = remove_column(nba_dna_matches, "Rk")
        nba_dna_matches = remove_column(nba_dna_matches, "G")
        nba_dna_matches = remove_column(nba_dna_matches, "GS")
        nba_dna_matches = move_column(nba_dna_matches, "PTS", 7)

        first_nba_match = extract_first_row(nba_dna_matches)

        return nba_dna_matches, first_nba_match
    else:
        logger.error("No NBA matches found.")
        return None, None

def compile_html_data(first_nba_match, college_latest_stats, college_player_stats_df, college_player_name, scrape_nba_player_data):
    nba_match_player_name = first_nba_match["Player"].iloc[0]
    nba_image_link, nba_player_height = scrape_nba_player_data(nba_match_player_name)
    
    nba_player_position = first_nba_match["Pos"].iloc[0]
    dna_match_percentage = first_nba_match["Similarity (%)"].iloc[0]
    nba_match_player_year = first_nba_match["Season"].iloc[0]

    dna_match_percentage = dna_match_percentage[:-1]  
    dna_match_percentage = float(dna_match_percentage)

    college_player_year = college_latest_stats['Season']

    college_latest_stats_df = pd.DataFrame([college_latest_stats], columns=college_player_stats_df.columns)
    college_stats_to_merge = college_latest_stats_df.head(1)
    nba_stats_to_merge = first_nba_match.head(1)

    logger.info("College Row: ")
    logger.info(college_latest_stats_df)
    logger.info("NBA Row: ")
    logger.info(nba_stats_to_merge)

    comparison_df = pd.concat([college_stats_to_merge, nba_stats_to_merge], ignore_index=True)

    logger.info("Comparison DF: ")
    logger.info(comparison_df)

    comparison_df = remove_column(comparison_df, "Similarity (%)")
    comparison_df = remove_column(comparison_df, "G")
    comparison_df = remove_column(comparison_df, "GS")
    comparison_df = remove_column(comparison_df, "Age")

    comparison_df.at[0, 'Player'] = college_player_name

    comparison_df = shift_df_col(comparison_df, 'Player', 1)
    comparison_df = shift_df_col(comparison_df, 'PTS', 7)

    return {
        "nba_match_player_name": nba_match_player_name,
        "nba_image_link": nba_image_link,
        "nba_player_height": nba_player_height,
        "nba_player_position": nba_player_position,
        "dna_match_percentage": dna_match_percentage,
        "nba_match_player_year": nba_match_player_year,
        "college_player_year": college_player_year,
        "comparison_df": comparison_df
    }