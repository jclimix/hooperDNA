from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hooper_dna():
    # Sample variables
    college_player = {
        "name": "John Doe",
        "position": "Guard",
        "height": "6'3\"",
        "points": 21.4,
        "assists": 3.4,
        "rebounds": 1.1
    }
    nba_player = {
        "name": "NBA Star",
        "position": "Guard",
        "height": "6'4\"",
        "points": 20.9,
        "assists": 3.8,
        "rebounds": 0.8
    }
    match_percentage = 96
    top_matches = [
        {"name": "Player 1", "match_percentage": 96, "stats": "..."},
        {"name": "Player 2", "match_percentage": 94, "stats": "..."},
        {"name": "Player 3", "match_percentage": 92, "stats": "..."}
    ]

    return render_template('hooper_dna.html',
                           college_player=college_player,
                           nba_player=nba_player,
                           match_percentage=match_percentage,
                           top_matches=top_matches)

if __name__ == '__main__':
    app.run(debug=True)
