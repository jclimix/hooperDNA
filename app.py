from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from loguru import logger
from waitress import serve
import pandas as pd
from loguru import logger

from utils import (
    get_college_player_name, 
    scrape_college_data, 
    adjust_stats, 
    create_weights_df,
    find_matches_before_college_player, 
    scrape_nba_player_data,
    round_stats
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# 500 Error Handler
@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html'), 500

@app.route('/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('sounds', filename)
        
@app.route('/submit', methods=['POST'])
def submit():
    
    logger.info("Processing form submission...")
    player_id = request.form.get('player_id')
    selected_profile = request.form.get('selected_profile')
    selected_algo = request.form.get('selected_algo')
    if not player_id or not selected_profile:

        logger.error('Missing player ID or profile selection. Redirecting to homepage.')
        return redirect(url_for('results'))
    
    logger.info("Player id & selected profile captured successfully...")
    return redirect(url_for('results', player_id=player_id, selected_profile=selected_profile, selected_algo=selected_algo))
        
@app.route('/results')
def results():
        
    college_player_id = request.args.get('player_id')
    selected_profile = request.args.get('selected_profile')
    selected_algo = request.args.get('selected_algo')

    logger.info(college_player_id)
    logger.info(selected_profile)
    logger.info(selected_algo)

    college_player_name = get_college_player_name(college_player_id)

    college_player_height, college_player_stats_df, college_headshot_link = scrape_college_data(college_player_id)

    adjusted_college_stats_df = adjust_stats(college_player_stats_df)

    college_player_season = str(college_player_stats_df['Season'].values[0])
    college_player_position = str(college_player_stats_df['Pos'].values[0])

    weights_df = create_weights_df(selected_profile)

    all_nba_matches = find_matches_before_college_player(college_player_season, adjusted_college_stats_df, weights_df, selected_algo)
    top_10_nba_matches = all_nba_matches.head(10)

    with pd.option_context('display.max_columns', None):
        logger.info(top_10_nba_matches)

    top_1_nba_match_name = top_10_nba_matches["Player"].iloc[0]
    top_1_nba_match_season = top_10_nba_matches["Season"].iloc[0]
    top_1_nba_match_position = top_10_nba_matches["Pos"].iloc[0]
    top_1_nba_match_dna_match_pct = top_10_nba_matches["DNA Match"].iloc[0]
    top_1_nba_match_stats = top_10_nba_matches.iloc[[0]]

    college_nba_join_stats = pd.concat([college_player_stats_df, top_1_nba_match_stats], axis=0, join='outer', ignore_index=True)
    columns_to_remove = ['Conf', 'Class', 'Rk', 'Player', 'Age', 'DNA Match']
    college_nba_join_stats = college_nba_join_stats.drop(columns=columns_to_remove)

    top_1_nba_match_headshot_link, top_1_nba_match_height = scrape_nba_player_data(top_1_nba_match_name)

    keep_columns = ['Season', 'Player', 'Team', 'Pos', 'Awards', 'DNA Match']
    mobile_top_10_nba_matches = top_10_nba_matches[keep_columns]

    logger.info(
        "NBA and College Player Match Info:\n"
        f"NBA Match Name: {top_1_nba_match_name}\n"
        f"NBA Match Season: {top_1_nba_match_season}\n"
        f"NBA Player Height: {top_1_nba_match_height}\n"
        f"NBA Match Link: {top_1_nba_match_headshot_link}\n"
        f"NBA Player Position: {top_1_nba_match_position}\n"
        f"DNA Match Percentage: {top_1_nba_match_dna_match_pct}\n"
        f"College Player Name: {college_player_name}\n"
        f"College Player Season: {college_player_season}\n"
        f"College Player Height: {college_player_height}\n"
        f"College Player Position: {college_player_position}\n"
        f"College Player Link: {college_headshot_link}"
    )

    # max values for stats on results page for bar chart:
    statbox_42 = 42.0  
    statbox_27 = 27.0 
    statbox_15 = 20.0 
    statbox_5 = 5.0 
    max_percentage = 1

    # stats that belong to each range:
    percentage_stats = ["FG%", "eFG%", "2P%", "3P%", "FT%"]
    statbox_42_stats = ["PTS", "MP"]
    statbox_27_stats = ["FG", "FGA", "3P", "3PA", "FT", "FTA"]
    statbox_15_stats = ["AST", "TRB", "ORB", "DRB"]
    statbox_5_stats = ["TOV", "PF", "STL", "BLK"]

    college_player_stats_df = round_stats(college_player_stats_df)
    top_1_nba_match_stats = round_stats(top_1_nba_match_stats)
    top_10_nba_matches = round_stats(top_10_nba_matches)
    mobile_top_10_nba_matches = round_stats(mobile_top_10_nba_matches)
    college_nba_join_stats = round_stats(college_nba_join_stats)

    selected_columns = percentage_stats + statbox_42_stats + statbox_27_stats + statbox_15_stats + statbox_5_stats
    existing_columns = [col for col in selected_columns if col in college_player_stats_df.columns]
    college_stats_filtered_df = college_player_stats_df[existing_columns]
    college_stats_dict = college_stats_filtered_df.to_dict(orient="records")[0]

    nba_stats_dict = {key: value for key, value in top_1_nba_match_stats.iloc[0].to_dict().items() if key in college_stats_dict}

    college_nba_join_stats = college_nba_join_stats.to_html(index=False)
    top_10_nba_matches = top_10_nba_matches.to_html(index=False)
    mobile_top_10_nba_matches = mobile_top_10_nba_matches.to_html(index=False)

    statbox_data = {
        'statbox_42': statbox_42,
        'statbox_27': statbox_27,
        'statbox_15': statbox_15,
        'statbox_5': statbox_5,
        'percentage_stats': percentage_stats,
        'statbox_42_stats': statbox_42_stats,
        'statbox_27_stats': statbox_27_stats,
        'statbox_15_stats': statbox_15_stats,
        'statbox_5_stats': statbox_5_stats,
        'max_percentage': max_percentage
    }

    nba_data = { 
            'name': top_1_nba_match_name,
            'image_link': top_1_nba_match_headshot_link,
            'height': top_1_nba_match_height,
            'position': top_1_nba_match_position,
            'stats': nba_stats_dict,
            'dna_match_percentage': top_1_nba_match_dna_match_pct,
            'match_year': top_1_nba_match_season
        }

    college_data = {
            'name': college_player_name,
            'height': college_player_height,
            'position': college_player_position,
            'image_link': college_headshot_link,
            'stats': college_player_stats_df,
            'player_year': college_player_season
        }

    return render_template(
                "comparison.html",
                nba_data=nba_data,
                college_data=college_data,
                college_stats=college_stats_dict,
                statbox_data=statbox_data,
                comparison_df=college_nba_join_stats,
                selected_profile=selected_profile,
                nba_dna_matches=top_10_nba_matches,
                mobile_top_10_nba_matches=mobile_top_10_nba_matches
            )

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8005)