import pandas as pd

player_name = 'First Last'

csv_file_path = './sample_DB/college_data/college_basketball_players.csv'
search_id = 'caitlin-clark-1'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Find the row where the 'ID' matches the search_id
row = df[df['playerId'] == search_id]

# If the row exists, return the value in the 'Player' column
if not row.empty:
    player_name = row['playerName'].values[0]

print(f"Player name for ID '{search_id}': {player_name}")
