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
import os, logging, boto3, pandas as pd, io
from waitress import serve

app = Flask(__name__)

load_dotenv()
logging.basicConfig(level=logging.INFO)

s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))


"""
import app
import cli-library
....
import z

MY_ENV_VAR = os.getenv(....)
...

def main():
    app.run_flask_app() <-- 
    it has all your catching of exceptions
    logic etc

if __name__ == '__main__':
    cli flag parsers
    click -- pkg
    typer
    main()

"""


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


def df_to_dict(df):

    return {col: df[col].values[0] for col in df.columns}


def shift_df_col(df, col, pos):
    column_to_move = df.pop(col)
    df.insert(pos, col, column_to_move)

    return df


def shift_dict_key(d, key, new_position):
    """
    Shifts the position of a key-value pair in a dictionary to a new position.

    Parameters:
    d (dict): The original dictionary.
    key (str): The key to be moved.
    new_position (int): The new position for the key (0-based index).

    Returns:
    dict: A new dictionary with the key-value pair moved to the new position.
    """
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


def scrape_nba_player_data(url):
    target_id = "meta"
    img_link = None
    height = None

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        target_section = soup.find(id=target_id)

        if not target_section:
            print(f"No section found with id '{target_id}'.")
        else:
            media_items = target_section.find_all("div", class_="media-item")

            if media_items:
                for index, item in enumerate(media_items, 1):
                    img_tag = item.find("img")
                    if img_tag and "src" in img_tag.attrs:
                        img_link = img_tag["src"]
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

        soup = BeautifulSoup(response.content, "html.parser")

        height_pattern = re.compile(r"([4-8]-\d{1,2})")

        height_element = soup.find("p", string=height_pattern)

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


def generate_json_from_csv():

    obj = s3.get_object(
        Bucket="hooperdna-storage", Key="college_data/college_basketball_players.csv"
    )
    csv_file = pd.read_csv(io.BytesIO(obj["Body"].read()))
    json_file = "./static/players.json"

    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    if not os.path.exists(json_file):
        data = []
        with open(csv_file, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({"name": row["playerName"], "id": row["playerId"]})

        with open(json_file, mode="w") as file:
            json.dump(data, file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():

    if request.method == "GET":
        print("see ya back home spiderman")
        return redirect(url_for("home"))

    print("i always get them...eventually")
    player_id = request.form.get("player_id")
    selected_profile = request.form.get("selected_profile")
    if not player_id or not selected_profile:

        print("Missing player ID or profile selection. Redirecting to homepage.")
        return redirect(url_for("home"))

    return redirect(
        url_for("results", player_id=player_id, selected_profile=selected_profile)
    )


@app.route("/results")
def results():
    # can you group any of this logic into generic functions?
    # "reusable" is important
    # do i read from s3 more than 1x? then it should probably be a function
    # do I do x > 1 consider making that a function -- and making it generic pasisng in variables

    # fluent
    # a().b().d().collect() ....
    college_player_id = request.args.get("player_id")
    selected_profile = request.args.get("selected_profile")

    # debugging
    print(college_player_id)
    print(selected_profile)

    obj = s3.get_object(
        Bucket="hooperdna-storage", Key="college_data/college_basketball_players.csv"
    )
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    row = df[df["playerId"] == college_player_id]

    if not row.empty:
        college_player_name = row["playerName"].values[0]

    # pull this out into it's own folder, so you import them
    # class PlayerStatWeights.college_player
    # class PlayerProfiles.weights
    # or put this not in a class but at the top of the application just outside as a global
    # if and only if you reference this data more than 1x

    # 1. get it working, which quick and dirty, shortcuts -- get the biz usecase solved as best you can
    # as you get more skilled quick and dirty -> really friggin good
    # 2. make it better, refactor, pay tech debt, write docs, etc
    # 3. can i find a project or refactor some existing code to do what i want, open source

    """
        user_data -> pull_corresponding_data_from_s3 -> update_player_dataframe -> establish_weights
         -> scrape_a_webpage_college_player ->
         -> compute_stats
         -> read_from s3 -> compute_player_stats
         -> comparison <-- all the value add is here

         calculations 

         if you come up with a standard table - dataframe - set of columns
         that's generic enough
         - player_name / id
         - team
         - year
         - ....
         - stats

         player_to_college() -> 10 college -> 12 for pro 1:1 college_stats * 12/10

         __eq__ == levestein similiarity analysis

         z_edey = df()
         bronny = df()
         z_edey == bronny true, false, null

         z_edey.similar_to(bronny) -> float         
          
        """

    college_player = {
        "MP": 0.0,
        "FG": 0.0,
        "FGA": 0.0,
        "FG%": 0.0,
        "3P": 0.0,
        "3PA": 0.0,
        "3P%": 0.0,
        "FT": 0.0,
        "FTA": 0.0,
        "FT%": 0.0,
        "ORB": 0.0,
        "DRB": 0.0,
        "TRB": 0.0,
        "AST": 0.0,
        "STL": 0.0,
        "BLK": 0.0,
        "TOV": 0.0,
        "PF": 0.0,
        "PTS": 0.0,
    }

    weight_profiles = {
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
        },
    }

    raw_weights = weight_profiles[selected_profile]

    total_weight = sum(raw_weights.values())
    #  what if total_weight was 0, error handling
    weights = {stat: value / total_weight for stat, value in raw_weights.items()}

    url = f"https://www.sports-reference.com/cbb/players/{college_player_id}.html"
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, "html.parser")
    div = soup.find("div", id="div_players_per_game")

    if div:
        table = div.find("table")
        """
            if condition:
               foo()
            else:
                log.error("log message")
                raise exception(....)
            """
        if table:
            college_player_stats_df = pd.read_html(str(table))[0]

            print(f"{college_player_name} | Stats:")  # logging
            print(college_player_stats_df)

            if "Season" in college_player_stats_df.columns:
                career_index = college_player_stats_df[
                    college_player_stats_df["Season"] == "Career"
                ].index

                if not career_index.empty:
                    latest_stats_index = career_index[0] - 1

                    if latest_stats_index >= 0:
                        latest_stats = college_player_stats_df.iloc[latest_stats_index]

                        for stat in college_player.keys():
                            if stat in latest_stats.index:
                                college_player[stat] = latest_stats[stat]

                        print(
                            f"\n{college_player_name}'s Statline for Euclidean Distance Analysis:"
                        )
                        print(
                            pd.DataFrame(
                                [latest_stats], columns=college_player_stats_df.columns
                            )
                        )

                        # College stat adjustments (NCAA => NBA)
                        college_player["MP"] *= 1  # 40 vs 48 total min
                        college_player[
                            "PTS"
                        ] *= 1  # skew scoring for better offensive player matches
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

    soup = BeautifulSoup(response.content, "html.parser")

    # Regular expression pattern to match heights (ranging from 4-0 to 8-11)
    height_pattern = re.compile(r"([4-8]-\d{1,2})")

    height_element = soup.find("span", string=height_pattern)

    if height_element:
        college_player_height = height_element.text.strip()

    else:
        print("Height element not found.")

    # pull college player headshot

    target_id = "meta"
    college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    target_section = soup.find(id=target_id)

    if not target_section:
        print(f"No section found with id '{target_id}'.")

    media_items = target_section.find_all("div", class_="media-item")

    if media_items:
        for index, item in enumerate(media_items, 1):
            img_tag = item.find("img")
            if img_tag and "src" in img_tag.attrs:
                college_image_link = img_tag["src"]

            else:
                print(f"Image {index}: No image found.")
                college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

    else:
        print(f"No media items found in the section with id '{target_id}'.")
        college_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"

    college_stats = np.array(
        [float(college_player[stat]) * weights[stat] for stat in college_player.keys()]
    ).reshape(1, -1)

    print("\nCustom Match Profile: " + str(selected_profile))  # f-string

    results = []

    # 2015 to 2024
    for year in range(2015, 2025):  # START_RANGE = 2015, END_RANGE = 2025
        obj = s3.get_object(
            Bucket="hooperdna-storage",
            Key=f"nba_raw_data/{year}_NBAPlayerStats_HprDNA_raw.csv",
        )
        csv_file = pd.read_csv(
            io.BytesIO(obj["Body"].read())
        )  # Use io.BytesIO to read the content from S3

        if not csv_file.empty:
            # query = f"SELECT * FROM '{csv_file}'"
            season_data = csv_file

            stat_columns = list(college_player.keys())

            if all(col in season_data.columns for col in stat_columns):
                nba_stats = season_data[stat_columns].fillna(0)

                college_player_position = latest_stats.get("Pos", "Unknown")

                pos_mapping = {"G": ["PG", "SG"], "F": ["SF", "PF"], "C": ["C"]}
                nba_positions = pos_mapping.get(college_player_position, [])

                if nba_positions:
                    filtered_season_data = season_data[
                        season_data["Pos"].isin(nba_positions)
                    ].reset_index(drop=True)
                    filtered_nba_stats = filtered_season_data[stat_columns].fillna(0)

                    weighted_nba_stats = filtered_nba_stats.apply(
                        lambda row: row
                        * np.array([weights[stat] for stat in stat_columns]),
                        axis=1,
                    )

                    distances = weighted_nba_stats.apply(
                        lambda row: euclidean(row, college_stats.flatten()), axis=1
                    )

                    distances_df = pd.DataFrame(
                        {"Distance": distances.values}, index=filtered_season_data.index
                    )

                    min_dist_index = np.argmin(distances)

                    if min_dist_index in distances_df.index:
                        distance_percentage = (
                            1 / (1 + distances_df.loc[min_dist_index, "Distance"])
                        ) * 100

                        most_similar_player_df = pd.DataFrame(
                            [filtered_season_data.iloc[min_dist_index]],
                            columns=filtered_season_data.columns,
                        )

                        most_similar_player_df.loc[
                            most_similar_player_df.index[0], "Similarity (%)"
                        ] = f"{distance_percentage:.2f}%"

                        results.append(most_similar_player_df)
                else:
                    print(
                        f"No matching NBA positions found for college position: {college_player_position}."
                    )
            else:
                print(
                    f"One or more columns from {stat_columns} are missing in CSV file."
                )
        else:
            print(f"File not found.")

    nba_dna_matches = pd.concat(results, ignore_index=True)

    nba_dna_matches = nba_dna_matches.sort_values(
        by="Similarity (%)", ascending=False
    ).reset_index(drop=True)

    nba_dna_matches = move_column(nba_dna_matches, "Similarity (%)", 1)
    nba_dna_matches = move_column(nba_dna_matches, "Player", 0)
    nba_dna_matches = remove_column(nba_dna_matches, "Rk")
    nba_dna_matches = remove_column(nba_dna_matches, "G")
    nba_dna_matches = remove_column(nba_dna_matches, "GS")
    nba_dna_matches = move_column(nba_dna_matches, "PTS", 7)

    first_nba_match = extract_first_row(nba_dna_matches)

    # more debugging
    print(f"\nBest NBA Player Match:")
    print(first_nba_match)

    print(f"\n{college_player_name}'s NBA Player Matches (In Last Decade):")
    print(nba_dna_matches)

    nba_match_player_name = first_nba_match["Player"].iloc[0]

    def find_nba_player_id(player_name, csv_file):
        obj = s3.get_object(
            Bucket="hooperdna-storage", Key=f"nba_player_data/nba_players_n_ids.csv"
        )
        df = pd.read_csv(
            io.BytesIO(obj["Body"].read())
        )  # Use io.BytesIO to read the content from S3

        player_row = df[df["playerName"].str.lower() == player_name.lower()]
        player_id = player_row["playerId"].values[0]
        return player_id

    nba_player_id = find_nba_player_id(nba_match_player_name, csv_file)
    first_char_nba_id = nba_player_id[0]

    nba_url = f"https://www.basketball-reference.com/players/{first_char_nba_id}/{nba_player_id}.html"

    nba_image_link = "https://i.ibb.co/vqkzb0m/temp-player-pic.png"
    nba_image_link, nba_player_height = scrape_nba_player_data(nba_url)

    nba_player_position = first_nba_match["Pos"].iloc[0]
    dna_match_percentage = first_nba_match["Similarity (%)"].iloc[0]
    nba_match_player_year = first_nba_match["Season"].iloc[0]

    dna_match_percentage = dna_match_percentage[:-1]
    dna_match_percentage = float(dna_match_percentage)

    college_player_year = latest_stats["Season"]

    latest_stats_df = pd.DataFrame(
        [latest_stats], columns=college_player_stats_df.columns
    )

    college_stats_to_merge = latest_stats_df.head(1)
    nba_stats_to_merge = first_nba_match.head(1)

    print("College Row: ")
    print(latest_stats_df)
    print("NBA Row: ")
    print(nba_stats_to_merge)
    comparison_df = pd.concat(
        [college_stats_to_merge, nba_stats_to_merge], ignore_index=True
    )

    print("Comparison DF: ")
    print(comparison_df)

    comparison_df = remove_column(comparison_df, "Similarity (%)")
    comparison_df = remove_column(comparison_df, "G")
    comparison_df = remove_column(comparison_df, "GS")
    comparison_df = remove_column(comparison_df, "Age")

    comparison_df.at[0, "Player"] = college_player_name

    comparison_df = shift_df_col(comparison_df, "Player", 1)
    comparison_df = shift_df_col(comparison_df, "PTS", 7)

    print("NBA Player Match Name: " + nba_match_player_name)
    print("NBA Player Match Season: " + nba_match_player_year)
    print("NBA Player Link: " + str(nba_image_link))
    print("NBA Player Height: " + (nba_player_height))
    print("NBA Player Position: " + (nba_player_position))
    print("DNA Match: " + str(dna_match_percentage))
    print("College Player Name: " + college_player_name)
    print("College Player Year: " + college_player_year)
    print("College Player Height: " + (college_player_height))
    print("College Player Position: " + (college_player_position))
    print("College Player Link: " + str(college_image_link))

    # Defining the max values for stats on results page for bar chart:
    statbox_42 = 42.0
    statbox_27 = 27.0
    statbox_15 = 15.0
    statbox_5 = 5.0
    max_percentage = 1

    # Defining which stats belong to which ranges:
    percentage_stats = ["FG%", "eFG%", "2P%", "3P%", "FT%"]
    statbox_42_stats = ["PTS", "MP"]
    statbox_27_stats = ["FG", "FGA", "3P", "3PA", "FT", "FTA"]
    statbox_15_stats = ["AST", "TRB", "ORB", "DRB"]
    statbox_5_stats = ["TOV", "PF", "STL", "BLK"]

    for key in college_player:
        college_player[key] = float(college_player[key])

    college_player = round_dict_values(college_player)
    first_nba_match = first_nba_match.round(2)

    first_nba_match = df_to_dict(first_nba_match)

    college_player = shift_dict_key(college_player, "PTS", 0)
    college_player = shift_dict_key(college_player, "AST", 1)
    college_player = shift_dict_key(college_player, "TRB", 2)
    college_player = shift_dict_key(college_player, "ORB", 3)
    college_player = shift_dict_key(college_player, "DRB", 4)
    college_player = shift_dict_key(college_player, "BLK", 5)
    college_player = shift_dict_key(college_player, "STL", 6)
    college_player = shift_dict_key(college_player, "FG%", 7)
    college_player = shift_dict_key(college_player, "FG", 8)
    college_player = shift_dict_key(college_player, "FGA", 9)
    college_player = shift_dict_key(college_player, "3P%", 10)
    college_player = shift_dict_key(college_player, "3P", 11)
    college_player = shift_dict_key(college_player, "3PA", 12)
    college_player = shift_dict_key(college_player, "FT%", 13)
    college_player = shift_dict_key(college_player, "FT", 14)
    college_player = shift_dict_key(college_player, "FTA", 15)
    college_player = shift_dict_key(college_player, "TOV", 16)
    college_player = shift_dict_key(college_player, "PF", 17)
    college_player = shift_dict_key(college_player, "MP", 18)

    print(
        college_player
    )  # proper logger loguru, important learn how the built python logging works
    print(first_nba_match)

    comparison_df = comparison_df.to_html(index=False)
    nba_dna_matches = nba_dna_matches.to_html(index=False)

    generate_json_from_csv()

    return render_template(
        "comparison.html",
        nba_match_player_name=nba_match_player_name,
        nba_image_link=nba_image_link,
        nba_player_height=nba_player_height,
        nba_player_position=nba_player_position,
        dna_match_percentage=dna_match_percentage,
        college_player_name=college_player_name,
        college_player_height=college_player_height,
        college_player_position=college_player_position,
        college_image_link=college_image_link,
        college_player=college_player,
        first_nba_match=first_nba_match,
        statbox_42=statbox_42,
        statbox_27=statbox_27,
        statbox_15=statbox_15,
        statbox_5=statbox_5,
        percentage_stats=percentage_stats,
        statbox_42_stats=statbox_42_stats,
        statbox_27_stats=statbox_27_stats,
        statbox_15_stats=statbox_15_stats,
        statbox_5_stats=statbox_5_stats,
        max_percentage=max_percentage,
        nba_match_player_year=nba_match_player_year,
        college_player_year=college_player_year,
        comparison_df=comparison_df,
        selected_profile=selected_profile,
        nba_dna_matches=nba_dna_matches,
    )


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8005)
