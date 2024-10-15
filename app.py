from flask import Flask, render_template, request, redirect, url_for
from loguru import logger
from waitress import serve

from utils import (
    read_csv_from_s3, 
    move_column, 
    remove_column, 
    extract_first_row, 
    round_dict_values, 
    df_to_dict, 
    shift_df_col, 
    shift_dict_key, 
    get_player_id, 
    scrape_nba_player_data, 
    generate_json_from_csv, 
    scrape_college_player_data, 
    find_nba_matches, 
    compile_html_data
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
        
@app.route('/submit', methods=['POST'])
def submit():

    if request.method == 'GET':
        logger.info('Redirecting to home page...')
        return redirect(url_for('home'))
    
    logger.info("Processing form submission...")
    player_id = request.form.get('player_id')
    selected_profile = request.form.get('selected_profile')
    if not player_id or not selected_profile:

        logger.error('Missing player ID or profile selection. Redirecting to homepage.')
        return redirect(url_for('home'))

    return redirect(url_for('results', player_id=player_id, selected_profile=selected_profile))
        
@app.route('/results')
def results():
        
        college_player_id = request.args.get('player_id')
        selected_profile = request.args.get('selected_profile')

        # debugging
        logger.info(college_player_id)
        logger.info(selected_profile)

        df = read_csv_from_s3('hooperdna-storage', 'college_data/college_basketball_players.csv')

        row = df[df["playerId"] == college_player_id]

        if not row.empty:
            college_player_name = row["playerName"].values[0]

        college_player = {"MP": 0.0, "FG": 0.0, "FGA": 0.0, "FG%": 0.0, "3P": 0.0, "3PA": 0.0, 
                        "3P%": 0.0, "FT": 0.0, "FTA": 0.0, "FT%": 0.0, "ORB": 0.0, "DRB": 0.0, 
                        "TRB": 0.0, "AST": 0.0, "STL": 0.0, "BLK": 0.0, "TOV": 0.0, "PF": 0.0, "PTS": 0.0}

        weight_profiles = {
            "offense": {"MP": 6.0, "FG": 7.0, "FGA": 5.0, "FG%": 6.0, "3P": 9.0, "3PA": 5.0, "3P%": 8.0,
                        "FT": 4.0, "FTA": 3.0, "FT%": 7.0, "ORB": 5.0, "DRB": 2.0, "TRB": 4.0, "AST": 7.0,
                        "STL": 4.0, "BLK": 4.0, "TOV": 3.0, "PF": 2.0, "PTS": 8.0},
            
            "defense": {"MP": 6.0, "FG": 4.0, "FGA": 3.0, "FG%": 5.0, "3P": 4.0, "3PA": 3.0, "3P%": 4.0,
                        "FT": 4.0, "FTA": 3.0, "FT%": 4.0, "ORB": 7.0, "DRB": 8.0, "TRB": 8.0, "AST": 5.0,
                        "STL": 9.0, "BLK": 9.0, "TOV": 2.0, "PF": 6.0, "PTS": 4.0},
            
            "balanced": {"MP": 7.0, "FG": 6.0, "FGA": 6.0, "FG%": 6.0, "3P": 6.0, "3PA": 6.0, "3P%": 7.0,
                        "FT": 6.0, "FTA": 6.0, "FT%": 6.0, "ORB": 6.0, "DRB": 6.0, "TRB": 6.0, "AST": 6.0,
                        "STL": 6.0, "BLK": 6.0, "TOV": 4.0, "PF": 5.0, "PTS": 6.0}
        }

        college_dataset = scrape_college_player_data(college_player_id, college_player, weight_profiles, selected_profile)

        college_player = college_dataset['stats']
        college_latest_stats = college_dataset['college_latest_stats']
        college_player_stats_df = college_dataset['college_player_stats_df']
        college_player_height = college_dataset['height']
        college_image_link = college_dataset['headshot']
        college_stats = college_dataset['weighted_stats']
        college_player_position = college_latest_stats.get("Pos", "Unknown")

        # cutoff for scraping college data
        
        nba_dna_matches, first_nba_match = find_nba_matches(college_player, college_latest_stats, college_stats, weight_profiles, selected_profile, read_csv_from_s3)

        # more debugging
        logger.info(f"\nBest NBA Player Match:")
        logger.info(first_nba_match)

        logger.info(f"\n{college_player_name}'s NBA Player Matches (In Last Decade):")
        logger.info(nba_dna_matches)

        html_data = compile_html_data(first_nba_match, college_latest_stats, college_player_stats_df, college_player_name, scrape_nba_player_data)

        nba_match_player_name = html_data["nba_match_player_name"]
        nba_image_link = html_data["nba_image_link"]
        nba_player_height = html_data["nba_player_height"]
        nba_player_position = html_data["nba_player_position"]
        dna_match_percentage = html_data["dna_match_percentage"]
        nba_match_player_year = html_data["nba_match_player_year"]
        college_player_year = html_data["college_player_year"]
        comparison_df = html_data["comparison_df"]

        logger.info(f"NBA Player Match Name: {nba_match_player_name}")
        logger.info(f"NBA Player Match Season: {nba_match_player_year}")
        logger.info(f"NBA Player Link: {nba_image_link}")
        logger.info(f"NBA Player Height: {nba_player_height}")
        logger.info(f"NBA Player Position: {nba_player_position}")
        logger.info(f"DNA Match: {dna_match_percentage}")
        logger.info(f"College Player Name: {college_player_name}")
        logger.info(f"College Player Year: {college_player_year}")
        logger.info(f"College Player Height: {college_player_height}")
        logger.info(f"College Player Position: {college_player_position}")
        logger.info(f"College Player Link: {college_image_link}")

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

        logger.info(college_player)
        logger.info(first_nba_match)

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
            nba_dna_matches=nba_dna_matches
        )

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8005)