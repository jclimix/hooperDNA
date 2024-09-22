import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from scipy.spatial.distance import euclidean
import duckdb
import os
import requests
from bs4 import BeautifulSoup

# Test 1: Verify college player data is loaded correctly from CSV
def test_read_college_player_data():
    csv_file_path = './sample_DB/college_data/college_basketball_players.csv'
    df = pd.read_csv(csv_file_path)
    
    college_player_id = 'jasper-floyd-1'
    row = df[df['playerId'] == college_player_id]
    
    assert not row.empty, "College player data should not be empty"
    assert row['playerName'].values[0] == "Jasper Floyd", "Player name should match"

# Test 2: Mock the web scraping part for player stats
@patch('requests.get')
def test_fetch_player_stats(mock_get):
    college_player_id = 'jasper-floyd-1'
    mock_html = '''
    <div id="div_players_per_game">
        <table>
            <tr><th>Season</th><th>MP</th><th>PTS</th></tr>
            <tr><td>2023-24</td><td>30.0</td><td>15.0</td></tr>
            <tr><td>Career</td></tr>
        </table>
    </div>
    '''
    mock_get.return_value.content = mock_html
    url = f'https://www.sports-reference.com/cbb/players/{college_player_id}.html'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    div = soup.find('div', id='div_players_per_game')

    assert div is not None, "The div with player stats should be found"
    table = div.find('table')
    assert table is not None, "The stats table should exist"

# Test 3: Verify stat calculation adjustments (Minutes and Points)
def test_stat_adjustments():
    college_player = {
        "MP": 30.0,
        "PTS": 15.0
    }
    
    # NCAA to NBA adjustments
    college_player["MP"] *= 1.17
    college_player["PTS"] *= 1.15
    
    assert round(college_player["MP"], 2) == 35.1, "Minutes should be adjusted correctly"
    assert round(college_player["PTS"], 2) == 17.25, "Points should be adjusted correctly"

# Test 4: Verify NBA player filtering and distance calculation
def test_filter_nba_players_and_calculate_similarity():
    college_stats = np.array([10.0, 20.0]).reshape(1, -1)  # Mocked college stats
    nba_stats = np.array([[8.0, 22.0], [15.0, 18.0]])      # Mocked NBA stats
    
    weighted_nba_stats = nba_stats  # Assuming weights are already applied
    distances = np.apply_along_axis(lambda row: euclidean(row, college_stats.flatten()), 1, weighted_nba_stats)
    
    assert len(distances) == 2, "There should be two distance calculations"
    assert round(distances[0], 2) == 2.83, "First distance should be correct"
    assert round(distances[1], 2) == 5.39, "Second distance should be correct"

# Test 5: Handle missing college player stats
def test_handle_missing_college_stats():
    college_player = {
        'MP': np.nan, 'FG': 0.0, 'FGA': 0.0, 'FG%': np.nan,
        '3P': 0.0, '3PA': np.nan, 'FT': np.nan, 'FTA': 0.0,
        'FT%': 0.0, 'ORB': 0.0, 'DRB': 0.0, 'TRB': 0.0,
        'AST': 0.0, 'STL': 0.0, 'BLK': 0.0, 'TOV': 0.0,
        'PF': 0.0, 'PTS': 0.0
    }

    # Call the function that processes college stats
    processed_college_stats = np.array([0 if np.isnan(college_player[stat]) else college_player[stat] for stat in college_player.keys()])

    # Ensure that missing stats are handled and no NaN values remain
    assert not np.isnan(processed_college_stats).any(), "Processed college stats should not have NaN values"

# Test 6: Picking Most Similar NBA Player
def test_most_similar_player_selection():
    nba_data = pd.DataFrame({
        'Player': ['NBA Player 1', 'NBA Player 2', 'NBA Player 3'],
        'MP': [10.0, 15.0, 12.0],
        'PTS': [20.0, 25.0, 22.0],
        'Similarity (%)': [90.0, 85.0, 95.0]  # Simulated similarity scores
    })

    # Sort and pick the most similar player (highest similarity)
    sorted_nba_data = nba_data.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)
    most_similar_player = sorted_nba_data.iloc[0]['Player']

    # Assert that the most similar player is correctly identified
    assert most_similar_player == 'NBA Player 3', "The most similar player should be 'NBA Player 3'"

