from unittest.mock import patch, Mock
from dotenv import load_dotenv
import logging, pandas as pd
import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv('../secrets/s3-hooperdna/.env')

from utils import (
    get_college_player_name, 
    scrape_college_data, 
    adjust_stats, 
    create_weights_df,
    calculate_dna_match,
    find_matches_before_college_player, 
    scrape_nba_player_data
)

@pytest.mark.parametrize("player_id, expected_result", [
    ("zach-edey-1", "Zach Edey"),
    ("caitlin-clark-1", "Caitlin Clark"),
    ("ryan-dunn-1", "Ryan Dunn")
    ])

def test_get_college_player_name(player_id, expected_result):

    result = get_college_player_name(player_id)

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize("player_id, expected_output", [
    ("zach-edey-1", ("7-4", pd.DataFrame(["PTS"], columns=["PTS"]), "https://www.sports-reference.com/req/202302071/cbb/images/players/zach-edey-1.jpg")),
    ("fake-id", (None, None, "https://i.ibb.co/vqkzb0m/temp-player-pic.png"))
])
def test_scrape_college_data(player_id, expected_output):
    result = scrape_college_data(player_id)

    expected_height, expected_df, expected_image_link = expected_output
    result_height, result_df, result_image_link = result

    assert result_height == expected_height, f"Expected height: {expected_height}, but got: {result_height}"

    if expected_df is not None:
        expected_columns = {"PTS", "AST", "TRB"}
        assert expected_columns.issubset(result_df.columns), f"Expected columns {expected_columns} in DataFrame, but they were not all found."
    else:
        assert result_df is None, "Expected None for DataFrame, but got a non-None value."


    assert result_image_link == expected_image_link, f"Expected image link: {expected_image_link}, but got: {result_image_link}"

def test_adjust_stats():
    data = {
        'MP': [20, 30, 25],
        'FG': [5, 10, 7],
        'PTS': [15, 22, 18]
    }
    df = pd.DataFrame(data)

    adjusted_df = adjust_stats(df)

    expected_data = {
        'MP': [22.6, 33.9, 28.25],
        'FG': [5.65, 11.3, 7.91],
        'PTS': [16.95, 24.86, 20.34]
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(adjusted_df, expected_df, check_exact=False, rtol=1e-3)


def test_create_weights_df():
    offense_weights = create_weights_df("offense")
    expected_offense_data = {
        "Stat": ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"],
        "Weight": [2.0, 6.0, 5.0, 6.5, 5.5, 4.5, 6.5, 5.0, 4.5, 5.5, 6.0, 5.0, 4.5, 5.5, 3.0, 2.5, 3.0, 6.0, 2.0, 2.0, 3.0, 2.0, 8.0]
    }
    expected_offense_df = pd.DataFrame(expected_offense_data).set_index("Stat")
    pd.testing.assert_frame_equal(offense_weights, expected_offense_df, check_exact=False, rtol=1e-3)

    defense_weights = create_weights_df("defense")
    expected_defense_data = {
        "Stat": ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"],
        "Weight": [2.0, 3.0, 2.5, 3.5, 2.5, 2.0, 3.0, 2.5, 2.0, 3.0, 3.5, 2.0, 1.5, 2.5, 5.5, 6.0, 6.5, 3.0, 6.5, 7.0, 5.0, 3.5, 3.0]
    }
    expected_defense_df = pd.DataFrame(expected_defense_data).set_index("Stat")
    pd.testing.assert_frame_equal(defense_weights, expected_defense_df, check_exact=False, rtol=1e-3)

    balanced_weights = create_weights_df("balanced")
    expected_balanced_data = {
        "Stat": ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"],
        "Weight": [3.0, 4.5, 4.0, 5.0, 4.0, 3.5, 5.0, 4.0, 3.5, 4.5, 5.0, 4.0, 3.5, 4.5, 4.0, 4.5, 5.0, 4.5, 4.0, 4.0, 3.5, 3.0, 5.5]
    }
    expected_balanced_df = pd.DataFrame(expected_balanced_data).set_index("Stat")
    pd.testing.assert_frame_equal(balanced_weights, expected_balanced_df, check_exact=False, rtol=1e-3)


def test_calculate_dna_match():
    college_player_data = {
        "Pos": ["G"],
        "MP": [20],
        "FG": [5],
        "PTS": [15]
    }
    nba_players_data = {
        "Pos": ["PG", "SG", "C"],
        "MP": [22, 18, 25],
        "FG": [6, 4, 8],
        "PTS": [17, 14, 20]
    }
    weights_data = {
        "Stat": ["MP", "FG", "PTS"],
        "Weight": [2.0, 6.0, 8.0]
    }

    college_player_df = pd.DataFrame(college_player_data)
    nba_players_df = pd.DataFrame(nba_players_data)
    weights_df = pd.DataFrame(weights_data).set_index("Stat")

    result_df = calculate_dna_match(college_player_df, nba_players_df, weights_df)

    assert "DNA Match" in result_df.columns

    assert result_df["DNA Match"].is_monotonic_decreasing

def test_find_matches_before_college_player(mocker):
    mock_load_nba_data = mocker.patch('utils.load_nba_data')
    mock_load_nba_data.side_effect = [
        pd.DataFrame({
            "Pos": ["PG", "SG"],
            "MP": [24, 30],
            "FG": [5, 7],
            "PTS": [16, 20],
            "DNA Match": [0, 0]  
        }) for _ in range(20) 
    ]

    college_player_data = {
        "Pos": ["G"],
        "MP": [20],
        "FG": [5],
        "PTS": [15]
    }
    college_player_df = pd.DataFrame(college_player_data)

    weights_data = {
        "Stat": ["MP", "FG", "PTS"],
        "Weight": [2.0, 6.0, 8.0]
    }
    weights_df = pd.DataFrame(weights_data).set_index("Stat")

    result_df = find_matches_before_college_player("2024", college_player_df, weights_df)

    assert "DNA Match" in result_df.columns
    assert result_df["DNA Match"].is_monotonic_decreasing

    assert not result_df.empty

def test_scrape_nba_player_data(mocker):
    mocker.patch('utils.read_csv_from_s3', return_value=pd.DataFrame({
        'playerName': ['Michael Jordan'],
        'playerId': ['d/michael-jordan-1']
    }))

    mock_response = mocker.Mock()
    mock_response.text = """
    <div id="meta">
        <div class="media-item"><img src="https://www.sports-reference.com/req/202302071/cbb/images/players/michael-jordan-1.jpg"/></div>
        <p>Height: 6-6</p>
    </div>
    """
    mock_response.status_code = 200
    mocker.patch('requests.get', return_value=mock_response)

    image_link, height = scrape_nba_player_data('Michael Jordan')

    assert image_link == "https://www.sports-reference.com/req/202302071/cbb/images/players/michael-jordan-1.jpg"
    assert height == "6-6"
