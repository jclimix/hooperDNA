import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
import os, logging, boto3, pandas as pd, io
from loguru import logger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv('../secrets/s3-hooperdna/.env')

from utils import (
    read_csv_from_s3,
    get_college_player_name, 
    scrape_college_data, 
    adjust_stats, 
    create_weights_df,
    find_matches_before_college_player, 
    scrape_nba_player_data
)

def test_get_college_player_name():

    player_id = "zach-edey-1"
    expected_result = "Zach Edey"

    result = get_college_player_name(player_id, expected_result)

    df = read_csv_from_s3('hooperdna-storage', 'college_data/college_basketball_players.csv')
    row = df[df["playerId"] == player_id]
    result = row["playerName"].values[0] if not row.empty else None

    assert result == expected_result, f"Expected {expected_result}, but got {result}"