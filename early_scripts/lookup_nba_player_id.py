import pandas as pd

def find_nba_player_id(player_name, csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Search for the player's ID based on their name
    player_row = df[df['playerName'].str.lower() == player_name.lower()]
    
    if not player_row.empty:
        player_id = player_row['playerId'].values[0]
        print(f"Player ID for '{player_name}': {player_id}")
    else:
        print(f"Player '{player_name}' not found.")

# Example usage
csv_file = './sample_DB/nba_player_data/nba_players_n_ids.csv'  # Replace with your CSV file path
player_name = 'LeBron James'
find_nba_player_id(player_name, csv_file)
