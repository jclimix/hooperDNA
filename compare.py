from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def compare_players():
    college_player_stats = {
        "PTS": 29.5, "AST": 5.2, "REB": 3.6, "BLK": 2.6, "STL": 1.9,
        "FG%": 49.5, "eFG%": 53.2, "2P%": 52.3, "3P%": 36.4, "FT%": 86.7
    }
    nba_player_stats = {
        "PTS": 30.4, "AST": 4.8, "REB": 3.8, "BLK": 2.5, "STL": 1.4,
        "FG%": 47.8, "eFG%": 52.1, "2P%": 50.4, "3P%": 38.2, "FT%": 88.1
    }
    
    # Define the max values for percentage and non-percentage stats
    max_pts_percentage = 29  # Set maximum value for non-percentage stats
    max_non_percentage = 15  # Set maximum value for non-percentage stats
    max_percentage = 100     # Maximum for percentage stats

    # Define the percentage stats to distinguish between the two ranges
    percentage_stats = ["FG%", "eFG%", "2P%", "3P%", "FT%"]
    pts_stats = ["PTS"]

    return render_template(
        'comparison.html', 
        college_player_stats=college_player_stats, 
        nba_player_stats=nba_player_stats,
        max_pts_percentage=max_pts_percentage, 
        max_percentage=max_percentage, 
        max_non_percentage=max_non_percentage,
        percentage_stats=percentage_stats,
        pts_stats=pts_stats
    )

if __name__ == '__main__':
    app.run(debug=True)
