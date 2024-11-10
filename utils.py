import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
import os, logging, boto3, pandas as pd, io
from loguru import logger
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HooperException(Execption):
    pass

# Inheritance method for swapping algos

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

# Base class for DNA match calculators
class BaseDNAMatchCalculator(ABC):
    @abstractmethod
    def calculate_dna_match(self, college_player_df: pd.DataFrame, nba_players_df: pd.DataFrame, weights_df: pd.DataFrame):
        pass

'''
class Foo:
    # class method because function does not depend on any attributes or data instinsic to the class
    # all of the logic is in the method
    @classmethod
    def bar():
        pass

/lib
/lib/matching.py
in lib/matching.py this function exist

from lib.matching import calculate_dna_match

if the CollegePlayer contained the weights to convert it's stats to NBA stats
''     NBAPlayer class contained the weights to convert it's stats to College stats

class NBAPlayer:
    def __init__(self, id: int, weights: df.DataFrame =predefined_df, raw_player_stats: df.DataFrame = foo):
        .... set stuff inside here

    def to_college():
        # lebron_college = NBAPlayer(1).to_college() == random_college_player() where this is a college player instance of a class

def calculate_dna_match(self, CollegePlayer, NBAPlayer):
    pass
    
class Player(ABC):
@abstractmethod
def transform():
because we have a binary choice of nba or college
based on teh class we're on
we will take the data stored in self.data and multiply or do whatever to it with the weights to convert to the other class's
this would return either a dataframe... for now

@abstractmethod
def calculate_player_distance(player_b):
pass



class NBAPlayer(Player):
each class has weights
def calculate_player_distance(player_b):
if isinstance(player_b) == NBAPlayer:
   then you're comparing two NBAplayer
   execute matching for two nba players
if isinstance(player_b)== CollegePlayer:
   then convert player_b to nbaplayer by calling transform
   then you can do fancy matching logic on nbaplayer.data_frame_containing_stats with player_b.transform()
else isinstance(player_b) not in [NBAPlayer, CollegePlayer]:
raise exception because what are we even doing here?

class CollegePlayer(Player):
each class has weights
def calculate_player_distance(player_b):
pass

nba_player_a

college_player_b

nba_player_a.calculate_player_distance(college_player_b)

global weights

weights = {
'college':
}

let's say we're convert from fahrenheit to celsious

conversion = {
'nba_to_college':function(),
'nba_to_g_leauge':function(),
'...'
}


how do i make this more generic and composable

currently too tightly coupled

i the programmer need to be intimately aware of the dataframes and their columns

but i would like them to be abstracted such that i can convert back and forth

pd.csv() -> it's all in the DataFrameClass that the csv function uses
pd.csv(file='foo.csv') 

'''

# Simple DNA match calculator class
class SimpleDNAMatchCalculator(BaseDNAMatchCalculator):
    def calculate_dna_match(self, college_player_df, nba_players_df, weights_df):
        try:
            if len(college_player_df) != 1:
                raise ValueError("college_player_df should contain only one row.")
            
            logger.info("Starting Simple DNA match calculation...")
            
            position_mapping = {
                "G": ["PG", "SG"],
                "F": ["SF", "PF"],
                "C": ["C"]
            }
            
            college_position = college_player_df.iloc[0]["Pos"]
            if college_position not in position_mapping:
                raise ValueError("Invalid college position")

            valid_nba_positions = position_mapping[college_position]
            nba_filtered_df = nba_players_df[nba_players_df["Pos"].isin(valid_nba_positions)].copy()
            
            logger.info(f"Filtered NBA DataFrame to {len(nba_filtered_df)} rows based on position.")

            stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
            logger.info(f"Using stat columns: {stat_columns}")

            filtered_weights = weights_df.loc[stat_columns].values.flatten()

            if np.linalg.norm(filtered_weights) != 0:
                filtered_weights /= np.linalg.norm(filtered_weights)

            college_stats = college_player_df[stat_columns].values.flatten().astype(float)

            dna_matches = []
            for idx, nba_row in nba_filtered_df.iterrows():
                nba_stats = nba_row[stat_columns].values.flatten().astype(float)

                valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)
                
                if not valid_indices.any():
                    logger.debug(f"Skipping row {idx} due to no valid indices.")
                    dna_matches.append(np.nan)
                    continue
                
                college_stats_valid = college_stats[valid_indices]
                nba_stats_valid = nba_stats[valid_indices]
                weights_valid = filtered_weights[valid_indices]

                weighted_college_stats = college_stats_valid * weights_valid
                weighted_nba_stats = nba_stats_valid * weights_valid

                dot_product = np.dot(weighted_college_stats, weighted_nba_stats)
                norm_college = np.linalg.norm(weighted_college_stats)
                norm_nba = np.linalg.norm(weighted_nba_stats)

                if norm_college == 0 or norm_nba == 0:
                    logger.debug(f"Skipping row {idx} due to zero norm.")
                    dna_matches.append(np.nan)
                    continue

                cosine_similarity = dot_product / (norm_college * norm_nba)

                absolute_differences = np.abs(weighted_college_stats - weighted_nba_stats)
                penalty_factor = min(1, np.mean(absolute_differences) / np.max(absolute_differences)) if np.max(absolute_differences) != 0 else 0

                adjusted_similarity_score = cosine_similarity * (1 - penalty_factor)
                similarity_score = round(adjusted_similarity_score * 100, 1)
                similarity_score = max(0, min(100, similarity_score))
                
                dna_matches.append(similarity_score)

            if len(dna_matches) != len(nba_filtered_df):
                logger.error("Mismatch in length between DNA matches and NBA DataFrame.")
                raise ValueError("Mismatch in length between DNA matches and NBA DataFrame.")

            nba_filtered_df["DNA Match"] = dna_matches
            logger.info("Simple DNA match calculation completed successfully.")
            return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

        # A catch all exception is often a code-smell
        except Exception as e:
            logger.error(f"An error occurred in SimpleDNAMatchCalculator: {e}")
            raise

# Legacy DNA match calculator class
class LegacyDNAMatchCalculator(BaseDNAMatchCalculator):
    def calculate_dna_match(self, college_player_df, nba_players_df, weights_df):
        try:
            if len(college_player_df) != 1:
                raise ValueError("college_player_df should contain only one row.")
            
            logger.info("Starting Legacy DNA match calculation...")
            
            position_mapping = {
                "G": ["PG", "SG"],
                "F": ["SF", "PF"],
                "C": ["C"]
            }
            
            college_position = college_player_df.iloc[0]["Pos"]
            if college_position not in position_mapping:
                raise ValueError("Invalid college position")

            valid_nba_positions = position_mapping[college_position]
            nba_filtered_df = nba_players_df[nba_players_df["Pos"].isin(valid_nba_positions)].copy()
            
            # this line here is too clever by half
            # consider a comment or a few intermediate steps
            stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
            logger.info(f"Using stat columns: {stat_columns}")

            filtered_weights = weights_df.loc[stat_columns].values.flatten()
            college_stats = college_player_df[stat_columns].values.flatten().astype(float)

            dna_matches = []
            for idx, nba_row in nba_filtered_df.iterrows():
                nba_stats = nba_row[stat_columns].values.flatten().astype(float)

                valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)
                
                if not valid_indices.any():
                    dna_matches.append(np.nan)
                    continue
                
                college_stats_valid = college_stats[valid_indices]
                nba_stats_valid = nba_stats[valid_indices]
                weights_valid = filtered_weights[valid_indices]

                weighted_diff = (college_stats_valid - nba_stats_valid) * weights_valid
                distance = np.linalg.norm(weighted_diff)
                
                max_distance = np.sqrt(len(stat_columns)) * np.max([np.ptp(college_stats_valid * weights_valid), np.ptp(nba_stats_valid * weights_valid)])
                max_distance = max_distance if max_distance != 0 else 1
                
                similarity_score = 100 * (1 - (distance / max_distance))
                similarity_score = round(similarity_score, 1)
                similarity_score = max(0, min(100, similarity_score))
                
                dna_matches.append(similarity_score)

            if len(dna_matches) != len(nba_filtered_df):
                logger.error("Mismatch in length between DNA matches and NBA DataFrame.")
                raise ValueError("Mismatch in length between DNA matches and NBA DataFrame.")

            nba_filtered_df["DNA Match"] = dna_matches
            logger.info("Legacy DNA match calculation completed successfully.")
            return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

        except Exception as e:
            logger.error(f"An error occurred in LegacyDNAMatchCalculator: {e}")
            raise HooperException("something insane happened")
            '''
            an exception was raise HooperException at line x message: "something insane happened"
            but you can see if the exeption had a good message and name it would add more context to the error
            good examples are the DivideByZero exception -- which tells you not that that something bad happened
            but what happened....
            '''

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

        if selected_algo == 'simple':

            simple_calculator = SimpleDNAMatchCalculator()
            nba_with_dna_match = simple_calculator.calculate_dna_match(adjusted_college_stats_df, nba_players_df, weights_df)
        
        elif selected_algo == 'legacy':

            legacy_calculator = LegacyDNAMatchCalculator()
            nba_with_dna_match = legacy_calculator.calculate_dna_match(adjusted_college_stats_df, nba_players_df, weights_df)
        
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
