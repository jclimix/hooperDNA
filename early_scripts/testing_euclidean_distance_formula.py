import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Sample DataFrame with 8 players and their stats, plus one new player
data = {
    'Player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8'],
    'PPG': [25.1, 22.3, 20.5, 24.8, 25.0, 21.1, 23.5, 18.9],
    'APG': [5.5, 6.2, 5.8, 6.0, 4.9, 5.6, 6.1, 5.3],
    'RPG': [7.2, 6.8, 7.0, 6.9, 6.7, 7.1, 6.9, 6.5],
    'FG%': [45.2, 44.5, 46.1, 45.8, 43.6, 44.0, 46.3, 42.8],
    '3P%': [37.0, 35.8, 36.5, 38.0, 34.9, 36.0, 37.5, 34.2],
    'MPG': [32.5, 30.0, 29.5, 31.0, 32.0, 30.5, 31.5, 27.5]  # Added Minutes per Game (MPG) stats
}

df = pd.DataFrame(data)

# Define weights for the stats, including MPG
weights = {
    'PPG': 1.2,  # Example of adjusted weights
    'APG': 1.0,
    'RPG': 1.0,
    'FG%': 1.0,
    '3P%': 1.0,
    'MPG': 1.2  # Example weight for MPG
}

# Apply the weights to the stats
df_weighted = df.copy()
for stat, weight in weights.items():
    df_weighted[stat] = df[stat] * weight

# Add a new player with a 7 in every stat, including MPG
new_player = pd.DataFrame([{
    'Player': 'Player9',
    'PPG': 7,
    'APG': 7,
    'RPG': 7,
    'FG%': 7,
    '3P%': 7,
    'MPG': 7
}])

# Append the new player to the existing DataFrame
df = pd.concat([df, new_player], ignore_index=True)

# Apply weights to the new player as well
df_weighted = df.copy()
for stat, weight in weights.items():
    df_weighted[stat] = df[stat] * weight

# Calculate Euclidean Distance matrix using weighted stats
euclidean_dist_matrix = euclidean_distances(df_weighted[['PPG', 'APG', 'RPG', 'FG%', '3P%', 'MPG']])

# Assuming you want to use Player1 as the target player
target_player_idx = df[df['Player'] == 'Player1'].index[0]

# Extract distance scores for the target player with others
distance_scores = euclidean_dist_matrix[target_player_idx]

# Convert distances to match percentages
min_distance = distance_scores.min()
max_distance = distance_scores.max()
match_percentages = 100 * (1 - (distance_scores - min_distance) / (max_distance - min_distance))

# Create a DataFrame to show stats and distance metrics
distance_df = pd.DataFrame({
    'Player': df['Player'],
    'PPG': df['PPG'],
    'APG': df['APG'],
    'RPG': df['RPG'],
    'FG%': df['FG%'],
    '3P%': df['3P%'],
    'MPG': df['MPG'],
    'Euclidean Distance': distance_scores,
    'Match Percentage': match_percentages
})

# Sort players by match percentage to Player1 (highest match first)
distance_df_sorted = distance_df.sort_values(by='Match Percentage', ascending=False)

print("\nMatch Percentage of Player1 to All Other Players (Including Stats):")
print(distance_df_sorted)
