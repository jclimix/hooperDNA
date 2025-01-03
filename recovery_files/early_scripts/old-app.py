from flask import Flask, render_template, request, redirect, url_for
from loguru import logger
from waitress import serve

from utils import (
    read_csv_from_s3,  
    round_dict_values, 
    df_to_dict, 
    shift_dict_key, 
    get_college_player_name,
    csv_to_dict,
    csv_to_nested_dict, 
    scrape_nba_player_data, 
    generate_json_from_csv, 
    scrape_college_player_data, 
    find_nba_matches, 
    compile_html_data
)

# capitalize global vars
college_player_csv = './csv_files/college_player.csv'
weight_profiles_csv = './csv_files/weight_profiles.csv'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
        
@app.route('/submit', methods=['POST'])
def submit():
    
    logger.info("Processing form submission...")
    player_id = request.form.get('player_id')
    selected_profile = request.form.get('selected_profile')
    if not player_id or not selected_profile:

        logger.error('Missing player ID or profile selection. Redirecting to homepage.')
        return redirect(url_for('results'))
    
    logger.info("Player id & selected profile captured successfully...")
    return redirect(url_for('results', player_id=player_id, selected_profile=selected_profile))
        
@app.route('/results')
def results():
        
        college_player_id = request.args.get('player_id')
        selected_profile = request.args.get('selected_profile')

        logger.info(college_player_id)
        logger.info(selected_profile)

        #refactor into function/mod to be called
        # for homme, submmit, result routes

        college_player_name = get_college_player_name(college_player_id)

        college_player = csv_to_dict(college_player_csv)
        weight_profiles = csv_to_nested_dict(weight_profiles_csv, key_column='profile')

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

        logger.info("NBA and College Player Match Info", {
            "NBA Match Name": nba_match_player_name,
            "NBA Match Season": nba_match_player_year,
            "NBA Match Link": nba_image_link,
            "NBA Player Height": nba_player_height,
            "NBA Player Position": nba_player_position,
            "DNA Match Percentage": dna_match_percentage,
            "College Player Name": college_player_name,
            "College Player Year": college_player_year,
            "College Player Height": college_player_height,
            "College Player Position": college_player_position,
            "College Player Link": college_image_link
        })


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

        stat_key_order = [
            "PTS", "AST", "TRB", "ORB", "DRB", "BLK", "STL", 
            "FG%", "FG", "FGA", "3P%", "3P", "3PA", "FT%", 
            "FT", "FTA", "TOV", "PF", "MP"
        ]

        for i, key in enumerate(stat_key_order):
            college_player = shift_dict_key(college_player, key, i)

        logger.info(college_player)
        logger.info(first_nba_match)

        comparison_df = comparison_df.to_html(index=False)
        nba_dna_matches = nba_dna_matches.to_html(index=False)

        generate_json_from_csv()

        logger.info("Script complete. Should open 'comparison.html'")

        nba_data = { 
            'name': nba_match_player_name,
            'image_link': nba_image_link,
            'height': nba_player_height,
            'position': nba_player_position,
            'stats': first_nba_match,
            'dna_match_percentage': dna_match_percentage,
            'match_year': nba_match_player_year
        }

        college_data = {
            'name': college_player_name,
            'height': college_player_height,
            'position': college_player_position,
            'image_link': college_image_link,
            'stats': college_player,
            'player_year': college_player_year
        }

        print(f'NBA Player Name: {college_data.name}')

        statbox_data = {
            '42': statbox_42,
            '27': statbox_27,
            '15': statbox_15,
            '5': statbox_5,
            'percentage_stats': percentage_stats,
            '42_stats': statbox_42_stats,
            '27_stats': statbox_27_stats,
            '15_stats': statbox_15_stats,
            '5_stats': statbox_5_stats,
            'max_percentage': max_percentage
        }

        return render_template(
            "comparison.html",
            nba_data=nba_data,
            college_data=college_data,
            statbox_data=statbox_data,
            comparison_df=comparison_df,
            selected_profile=selected_profile,
            nba_dna_matches=nba_dna_matches
        )


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8005)