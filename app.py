from flask import Flask, render_template, request, jsonify
import pandas as pd
from waitress import serve
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from dna_processor import (
    scrape_player, scrape_stats, scrape_player_height, get_college_player_name,
    rename_columns, simple_calculate_dna_match, create_weights_df,
    extract_table_to_df, season_to_year, year_to_season, get_last_10_seasons,
    filter_by_position, consolidate_traded_players, remove_duplicate_players,
    filter_games_played
)

def get_version():
    try:
        with open('version.txt', 'r') as f:
            version = f.read().strip()
        return version
    except Exception as e:
        print(f"Error reading version file: {e}")
        return "v?.?.?" 
    
college_player_ids_df = None

version = get_version()

def create_app():

    app = Flask(__name__, static_folder='static')
    
    with app.app_context():
        load_data()
    
    @app.route('/')
    def index():
        return render_template('index.html', version=version)
    
    @app.route('/search')
    def search():
        return render_template('search.html')

    # API endpoint for player name search
    @app.route('/api/search_players')
    def search_players():
        global college_player_ids_df
        query = request.args.get('q', '').lower()
        
        if not query or len(query) < 2:
            return jsonify({'players': []})
        
        # reload data if it's not loaded as fallback
        if college_player_ids_df is None:
            load_data()
        
        # filter college players by name
        filtered_players = college_player_ids_df[college_player_ids_df['player_name'].str.lower().str.contains(query)]
        
        # sort by relevance (exact match first, then startswith, then contains)
        exact_matches = filtered_players[filtered_players['player_name'].str.lower() == query]
        starts_with = filtered_players[filtered_players['player_name'].str.lower().str.startswith(query) & 
                                    ~filtered_players['player_name'].str.lower().isin(exact_matches['player_name'].str.lower())]
        contains = filtered_players[~filtered_players['player_name'].str.lower().isin(exact_matches['player_name'].str.lower()) & 
                                ~filtered_players['player_name'].str.lower().isin(starts_with['player_name'].str.lower())]
        
        # combine and limit to 10 results
        sorted_players = pd.concat([exact_matches, starts_with, contains]).head(10)
        
        # convert to list of dictionaries
        results = sorted_players.to_dict('records')
        
        return jsonify({'players': results})

    def safe_float_convert(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @app.route('/process', methods=['POST'])
    def process_dna_match():
        global college_player_ids_df
        
        try:
            college_player_id = request.form.get('player_id')
            algo_weight = request.form.get('algo_weight', 'offense')  # default to offense

            algo_weight_str = algo_weight.capitalize()
            
            if college_player_ids_df is None:
                load_data()
            
            # get college player data
            soup = scrape_player(college_player_id)
            college_player_name = get_college_player_name(college_player_ids_df, college_player_id)
            college_player_height = scrape_player_height(soup)
            college_player_stats_df = scrape_stats(soup)
            
            # generate college player image URL
            college_player_slug = college_player_name.lower().replace(' ', '-')
            college_image_url = f"https://www.sports-reference.com/req/202302071/cbb/images/players/{college_player_slug}-1.jpg"
            
            # get the latest season and position
            latest_season = college_player_stats_df["Season"].iloc[0]
            college_position = college_player_stats_df["Pos"].iloc[0]
            latest_year = season_to_year(latest_season)
            
            # get last 10 seasons for comparison
            last_10_list = get_last_10_seasons(latest_year)
            
            # get NBA stats for comparison
            nba_stats_dict = {year: extract_table_to_df(f'{year}_reg_season_stats', 'per_game_stats') for year in last_10_list}
            
            # process NBA stats
            for year in last_10_list:
                nba_stats_dict[year] = consolidate_traded_players(nba_stats_dict[year])
                nba_stats_dict[year] = filter_by_position(nba_stats_dict[year], college_position)
            
            college_player_stats_df = rename_columns(college_player_stats_df)
            
            # create weights for the algorithm
            df_weights = create_weights_df(algo_weight)
            final_matches_df = pd.DataFrame()
            
            # calc DNA matches for each NBA season
            for year in last_10_list:
                matches_df = simple_calculate_dna_match(college_player_stats_df, nba_stats_dict[year], df_weights)
                season = year_to_season(year)
                matches_df.insert(1, "season", [season for _ in range(len(matches_df))])
                matches_df = matches_df[:10]  # Keep top 10 from each season
                final_matches_df = pd.concat([final_matches_df, matches_df], ignore_index=True)
            
            # process and sort final matches
            final_matches_df = final_matches_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)
            final_matches_df = remove_duplicate_players(final_matches_df)
            final_matches_df = filter_games_played(final_matches_df)
            top_matches_df = final_matches_df[:3]  # Top 3 matches
            
            # convert stats to numeric values to avoid string issues
            for col in college_player_stats_df.columns:
                if col not in ['Season', 'Team', 'Conf', 'Pos', 'Class']:
                    college_player_stats_df[col] = pd.to_numeric(college_player_stats_df[col], errors='coerce')
            
            for col in top_matches_df.columns:
                if col not in ['player_name', 'player_id', 'team', 'position', 'season']:
                    top_matches_df[col] = pd.to_numeric(top_matches_df[col], errors='coerce')
            
            # format data for template with safe conversion
            college_stats_dict = college_player_stats_df.iloc[0].to_dict()
            for key, value in college_stats_dict.items():
                if key not in ['Season', 'Team', 'Conf', 'Pos', 'Class']:
                    college_stats_dict[key] = safe_float_convert(value)
            
            # list of all stats to ensure they exist in dictionary
            all_stats = ['turnovers', 'points', 'steals', 'free_throw_pct', 'three_point_pct', 
                        'field_goal_pct', 'offensive_rebounds', 'three_pointers_made', 
                        'personal_fouls', 'two_pointers_made', 'blocks', 'free_throws_made', 
                        'assists', 'two_point_pct', 'total_rebounds', 'defensive_rebounds', 
                        'minutes_played']
                        
            # ensure stats exist in dictionary (use 0 if missing)
            for stat in all_stats:
                if stat not in college_stats_dict:
                    college_stats_dict[stat] = 0.0
            
            college_player_data = {
                'name': college_player_name if college_player_name else "Unknown Player",
                'height': college_player_height if college_player_height else "Unknown",
                'position': college_position if college_position else "Unknown",
                'season': latest_season if latest_season else "Unknown",
                'stats': college_stats_dict,
                'image_url': college_image_url
            }
            
            # NBA matches data with safe conversion
            nba_matches = []
            for _, row in top_matches_df.iterrows():
                match_stats = row.to_dict()
                
                # check all required stats exist in the dictionary
                for stat in all_stats:
                    if stat not in match_stats:
                        match_stats[stat] = 0.0
                    else:
                        match_stats[stat] = safe_float_convert(match_stats[stat])
                
                # convert other numeric stats
                for key, value in match_stats.items():
                    if key not in ['player_name', 'player_id', 'team', 'position', 'season', 'DNA Match'] + all_stats:
                        match_stats[key] = safe_float_convert(value)
                
                # NBA player image URL
                nba_player_id = row['player_id'] if 'player_id' in row else ""
                nba_image_url = f"https://www.basketball-reference.com/req/202106291/images/headshots/{nba_player_id}.jpg"
                
                nba_matches.append({
                    'name': row['player_name'] if 'player_name' in row else "Unknown",
                    'team': row['team'] if 'team' in row else "Unknown",
                    'position': row['position'] if 'position' in row else "Unknown",
                    'season': row['season'] if 'season' in row else "Unknown",
                    'match_percentage': safe_float_convert(row['DNA Match']) if 'DNA Match' in row else 0.0,
                    'stats': match_stats,
                    'image_url': nba_image_url
                })
            
            # comparison data for charts with safe conversion
            comparison_labels = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'FG%', '3P%']
            
            # get with default value to handle missing keys
            college_data = [
                safe_float_convert(college_stats_dict.get('points', 0)),
                safe_float_convert(college_stats_dict.get('total_rebounds', 0)),
                safe_float_convert(college_stats_dict.get('assists', 0)),
                safe_float_convert(college_stats_dict.get('steals', 0)),
                safe_float_convert(college_stats_dict.get('blocks', 0)),
                safe_float_convert(college_stats_dict.get('field_goal_pct', 0) * 100),
                safe_float_convert(college_stats_dict.get('three_point_pct', 0) * 100)
            ]
            
            nba_data = []
            if nba_matches:
                top_match_stats = nba_matches[0]['stats']
                nba_data = [
                    safe_float_convert(top_match_stats.get('points', 0)),
                    safe_float_convert(top_match_stats.get('total_rebounds', 0)),
                    safe_float_convert(top_match_stats.get('assists', 0)),
                    safe_float_convert(top_match_stats.get('steals', 0)),
                    safe_float_convert(top_match_stats.get('blocks', 0)),
                    safe_float_convert(top_match_stats.get('field_goal_pct', 0) * 100),
                    safe_float_convert(top_match_stats.get('three_point_pct', 0) * 100)
                ]
            else:
                nba_data = [0, 0, 0, 0, 0, 0, 0]
            
            comparison_data = {
                'college_player': college_data,
                'nba_player': nba_data
            }
            
            return render_template(
                'comparison.html',
                college_player=college_player_data,
                nba_matches=nba_matches,
                comparison_labels=comparison_labels,
                comparison_data=comparison_data,
                algo_weight_str=algo_weight_str,
                version=version
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("Error details:", error_details)
            return render_template('error.html',
                                version=version,
                                error_title="Processing Error",
                                error_message="We encountered an error while processing your request.",
                                error_code="500"), 500

    # error handler for all error types
    @app.errorhandler(Exception)
    def handle_error(e):
        error_code = getattr(e, 'code', 500)
        error_title = "Error"
        error_message = "We encountered an error while processing your request."
        
        # message based on common error codes
        if error_code == 404:
            error_title = "Page Not Found"
            error_message = "The page you're looking for doesn't exist."
        elif error_code == 403:
            error_title = "Access Forbidden"
            error_message = "You don't have permission to access this resource."
        elif error_code == 405:
            error_title = "Method Not Allowed"
            error_message = "The method is not allowed for the requested URL."
        
        return render_template('error.html', 
                            error_title=error_title,
                            version=version,
                            error_message=error_message,
                            error_code=str(error_code)), error_code
    
    return app

def load_data():
    """Load the college player IDs data once at startup."""
    global college_player_ids_df
    college_player_ids_df = extract_table_to_df('college_player_ids', 'college')
    print(f"Loaded {len(college_player_ids_df)} college player records")

app = create_app()

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]
)

if __name__ == '__main__':
    
    # app.run(debug=True, port=8005)
    serve(app, host="0.0.0.0", port=8005)