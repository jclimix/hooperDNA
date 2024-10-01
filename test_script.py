import pytest
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock
from scipy.spatial.distance import euclidean

from main import (
    init_college_player_info, 
    init_weight_profiles, 
    scrape_college_stats, 
    find_nba_matches
)

# Sample player fixture
@pytest.fixture
def sample_college_player():
    return 'jasper-floyd-1'

# CSV file path fixture
@pytest.fixture
def csv_file_path():
    return './sample_DB/college_data/college_basketball_players.csv'

# Sample college player dictionary fixture
@pytest.fixture
def sample_college_player_dict():
    return {
        'MP': 10.0,
        'FG': 4.0,
        'FGA': 8.0,
        'FG%': 50.0,
        '3P': 2.0,
        '3PA': 5.0,
        '3P%': 40.0,
        'FT': 3.0,
        'FTA': 4.0,
        'FT%': 75.0,
        'ORB': 1.0,
        'DRB': 3.0,
        'TRB': 4.0,
        'AST': 5.0,
        'STL': 2.0,
        'BLK': 1.0,
        'TOV': 2.0,
        'PF': 3.0,
        'PTS': 12.0
    }

# Test for initializing college player information
def test_init_college_player_info(sample_college_player, csv_file_path):
    # Mocking the CSV data
    data = {
        'playerId': [sample_college_player],
        'playerName': ['Jasper Floyd']
    }
    df = pd.DataFrame(data)

    with patch('pandas.read_csv', return_value=df):
        player_name, player_dict = init_college_player_info(sample_college_player, csv_file_path)
        assert player_name == 'Jasper Floyd'
        assert isinstance(player_dict, dict)
        assert 'MP' in player_dict

# Test for initializing weight profiles
def test_init_weight_profiles():
    weights = init_weight_profiles('offense')
    assert isinstance(weights, dict)
    assert weights['MP'] > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-5

# Test for scraping college stats
def test_scrape_college_stats(sample_college_player, csv_file_path):
    # Step 1: Initialize player info
    college_player_name, college_player_dict = init_college_player_info(sample_college_player, csv_file_path)
    
    # Step 2: Scrape real data from the website
    college_latest_stats, updated_college_player_dict = scrape_college_stats(sample_college_player, college_player_name, college_player_dict)
    
    # Step 3: Assert that the stats have been successfully pulled and updated
    assert college_latest_stats is not None, "No stats were pulled from the website."
    assert updated_college_player_dict is not None, "The player dictionary was not updated with new stats."
    
    # Step 4: Check that at least some stats have non-zero values after scraping
    assert any(value != 0 for value in updated_college_player_dict.values()), "No stats were updated in the player dictionary."

# Test for finding NBA matches
def test_find_nba_matches(sample_college_player_dict):
    weights = init_weight_profiles('offense')

    # Mocking NBA data with 3 guards
    nba_data = pd.DataFrame({
        'MP': [30.0, 25.0, 28.0],
        'FG': [9.0, 8.0, 7.5],
        'FGA': [18.0, 16.0, 17.0],
        'FG%': [50.0, 50.0, 44.1],
        '3P': [3.0, 2.0, 2.5],
        '3PA': [7.0, 5.0, 6.0],
        '3P%': [42.9, 40.0, 41.7],
        'FT': [4.0, 3.0, 3.5],
        'FTA': [5.0, 4.0, 5.0],
        'FT%': [80.0, 75.0, 78.0],
        'ORB': [1.0, 0.5, 0.8],
        'DRB': [3.0, 4.0, 3.5],
        'TRB': [4.0, 4.5, 4.3],
        'AST': [6.0, 5.0, 4.5],
        'STL': [2.0, 1.5, 1.8],
        'BLK': [1.0, 0.8, 0.9],
        'TOV': [2.5, 2.0, 2.3],
        'PF': [3.0, 2.5, 2.8],
        'PTS': [25.0, 22.0, 23.5],
        'Pos': ['PG', 'PG', 'SG']
    })
    
    with patch('duckdb.query', return_value=MagicMock(df=lambda: nba_data)):
        nba_matches = find_nba_matches(sample_college_player_dict, weights, sample_college_player_dict)
        assert not nba_matches.empty
        assert 'Similarity (%)' in nba_matches.columns
