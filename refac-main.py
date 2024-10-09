from flask import Flask, render_template, request, redirect, url_for, jsonify
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
import logging, boto3, io

app = Flask(__name__)

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize S3 client using environment variables
s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))

# ========== Utility Functions ==========

def move_column(df, column_name, new_position):
    """Move a column to a new position in DataFrame."""
    column = df.pop(column_name)
    df.insert(new_position, column_name, column)
    return df

def remove_column(df, column_name):
    """Remove a column from DataFrame."""
    df = df.drop(columns=[column_name])
    return df

def extract_first_row(df):
    """Extract the first row of a DataFrame."""
    new_df = df.iloc[[0]].reset_index(drop=True)
    return new_df

def round_dict_values(input_dict):
    """Round all numeric values in a dictionary to 2 decimal places."""
    rounded_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, (int, float)):
            rounded_dict[key] = round(value, 2)
        else:
            rounded_dict[key] = value  # Keep non-numeric values unchanged
    return rounded_dict

def shift_dict_key(d, key, new_position):
    """Shift a key-value pair in a dictionary to a new position."""
    if key not in d:
        raise KeyError(f"Key '{key}' not found in dictionary.")
    key_value_pair = {key: d.pop(key)}
    items = list(d.items())
    items.insert(new_position, list(key_value_pair.items())[0])
    return dict(items)

# ========== Data Handling Functions ==========

def get_player_id(player_name, csv_file):
    """Retrieve the player ID from the player name."""
    df = pd.read_csv(csv_file)
    matching_players = df[df["playerName"].str.lower() == player_name.lower()]
    if not matching_players.empty:
        return matching_players.iloc[-1]["playerId"]
    return None

def read_s3_csv(bucket_name, key):
    """Read a CSV file from S3."""
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def scrape_nba_player_data(url):
    """Scrape NBA player data from the provided URL."""
    img_link, height = None, None
    try:
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        target_section = soup.find(id="meta")
        media_items = target_section.find_all("div", class_="media-item")
        img_link = media_items[0].find("img")["src"] if media_items else None
        
        height_element = soup.find("p", string=re.compile(r"([4-8]-\d{1,2})"))
        height = height_element.text.strip() if height_element else None
    except Exception as e:
        logging.error(f"Error scraping NBA player data: {e}")
    return img_link, height

def generate_json_from_csv():
    """Generate a JSON file from a CSV file stored in S3."""
    obj = s3.get_object(Bucket='hooperdna-storage', Key='college_data/college_basketball_players.csv')
    csv_file = pd.read_csv(io.BytesIO(obj['Body'].read()))
    json_file = './static/players.json'
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    if not os.path.exists(json_file):
        data = [{'name': row['playerName'], 'id': row['playerId']} for _, row in csv_file.iterrows()]
        with open(json_file, 'w') as file:
            json.dump(data, file)

# ========== Flask Routes ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission for player comparison."""
    if request.method == 'POST':
        player_id = request.form.get('player_id')
        selected_profile = request.form.get('selected_profile')
        if not player_id or not selected_profile:
            logging.warning('Missing player ID or profile selection. Redirecting to homepage.')
            return redirect(url_for('home'))
        return redirect(url_for('results', player_id=player_id, selected_profile=selected_profile))

@app.route('/results')
def results():
    """Display the comparison results between college and NBA players."""
    player_id = request.args.get('player_id')
    selected_profile = request.args.get('selected_profile')

    # Load college player data from S3
    df = read_s3_csv('hooperdna-storage', 'college_data/college_basketball_players.csv')
    college_player_row = df[df["playerId"] == player_id]

    if college_player_row.empty:
        logging.error(f"Player with ID {player_id} not found in the database.")
        return redirect(url_for('home'))

    college_player_name = college_player_row["playerName"].values[0]
    logging.info(f"Processing data for {college_player_name}...")

    # Process and return comparison data
    results_df = process_player_comparison(player_id, selected_profile)
    logging.info(f"Comparison results processed successfully for {college_player_name}.")
    
    return render_template("comparison.html", results=results_df)

# ========== Data Processing Functions ==========

def process_player_comparison(player_id, selected_profile):
    """Compare a college player with NBA players based on the selected profile."""
    weight_profiles = load_weight_profiles()
    raw_weights = weight_profiles[selected_profile]
    total_weight = sum(raw_weights.values())
    weights = {stat: value / total_weight for stat, value in raw_weights.items()}

    # Perform Euclidean distance comparison with NBA players
    nba_dna_matches = perform_comparison(player_id, weights)
    
    # Add debug statement to check the column names
    logging.info(f"Columns in nba_dna_matches: {nba_dna_matches.columns}")
    
    # Check if the 'Player' column exists before accessing it
    if 'Player' in nba_dna_matches.columns:
        best_match_player = nba_dna_matches.iloc[0]['Player']
        logging.info(f"Comparison complete. Best match: {best_match_player}")
    else:
        logging.warning("'Player' column not found in nba_dna_matches DataFrame.")
        best_match_player = "Unknown"
    
    return nba_dna_matches

def perform_comparison(player_id, weights):
    """Perform the comparison between college stats and NBA players' stats."""
    results = []
    for year in range(2015, 2025):
        df = read_s3_csv('hooperdna-storage', f'nba_raw_data/{year}_NBAPlayerStats_HprDNA_raw.csv')
        filtered_data = filter_nba_data(df, weights)
        best_match = find_best_nba_match(filtered_data, weights)
        results.append(best_match)

    return pd.concat(results, ignore_index=True)

def filter_nba_data(df, weights):
    """Filter NBA player data and calculate weighted stats."""
    stat_columns = weights.keys()
    return df[stat_columns].fillna(0)

def find_best_nba_match(df, weights):
    """Find the NBA player with the closest stats to the college player."""
    # Convert weights to a NumPy array
    weights_array = np.array(list(weights.values()))
    
    # Ensure that all rows and weights are 1-D arrays
    def calculate_distance(row):
        return euclidean(row.values.flatten(), weights_array)

    # Apply the function to calculate the distance for each row
    distances = df.apply(lambda row: calculate_distance(row), axis=1)
    
    # Find the index of the minimum distance
    min_dist_index = np.argmin(distances)
    
    # Debug the DataFrame to check available columns
    logging.debug(f"Available columns in DataFrame: {df.columns}")
    
    # Return the row with the closest match
    return df.iloc[[min_dist_index]]


def load_weight_profiles():
    """Load the weight profiles for different player types."""
    return {
        "offense": {
                "MP": 6.0,
                "FG": 7.0,
                "FGA": 5.0,
                "FG%": 6.0,
                "3P": 9.0,
                "3PA": 5.0,
                "3P%": 8.0,
                "FT": 4.0,
                "FTA": 3.0,
                "FT%": 7.0,
                "ORB": 5.0,
                "DRB": 2.0,
                "TRB": 4.0,
                "AST": 7.0,
                "STL": 4.0,
                "BLK": 4.0,
                "TOV": 3.0,
                "PF": 2.0,
                "PTS": 8.0,
            },
            "defense": {
                "MP": 6.0,
                "FG": 4.0,
                "FGA": 3.0,
                "FG%": 5.0,
                "3P": 4.0,
                "3PA": 3.0,
                "3P%": 4.0,
                "FT": 4.0,
                "FTA": 3.0,
                "FT%": 4.0,
                "ORB": 7.0,
                "DRB": 8.0,
                "TRB": 8.0,
                "AST": 5.0,
                "STL": 9.0,
                "BLK": 9.0,
                "TOV": 2.0,
                "PF": 6.0,
                "PTS": 4.0,
            },
            "balanced": {
                "MP": 7.0,
                "FG": 6.0,
                "FGA": 6.0,
                "FG%": 6.0,
                "3P": 6.0,
                "3PA": 6.0,
                "3P%": 7.0,
                "FT": 6.0,
                "FTA": 6.0,
                "FT%": 6.0,
                "ORB": 6.0,
                "DRB": 6.0,
                "TRB": 6.0,
                "AST": 6.0,
                "STL": 6.0,
                "BLK": 6.0,
                "TOV": 4.0,
                "PF": 5.0,
                "PTS": 6.0,
            }
    }

# ========== Testing ==========

if __name__ == "__main__":
    logging.info("Starting the Flask app in debug mode.")
    app.run(debug=True)