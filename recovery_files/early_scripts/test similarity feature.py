import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a DataFrame to hold player stats
data = {
    'Player': ['Player 1', 'Player 2', 'Player 3'],
    'FGM': [9.2, 10.1, 8.5],
    'FG%': [0.47, 0.52, 0.44],
    'PPG': [25.3, 27.5, 20.7],
    'APG': [5.6, 7.3, 6.1],
    'MPG': [38.0, 36.5, 35.1],
    '3FGM': [2.5, 3.1, 1.9],
    '3FG%': [0.38, 0.42, 0.34],
    '2FGM': [6.7, 7.0, 6.6],
    '2FG%': [0.52, 0.56, 0.48],
    'TRB': [11.2, 9.5, 7.8],
    'OREB': [2.1, 1.8, 1.6],
    'DREB': [9.1, 7.7, 6.2],
    'eFG%': [0.53, 0.56, 0.49],
    'FT%': [0.85, 0.88, 0.80],
    'FTM': [6.5, 5.7, 4.8],
    'TOV': [2.8, 3.1, 2.5],
    'SPG': [1.6, 1.8, 1.4],
    'BPG': [0.9, 1.1, 0.7]
}

df = pd.DataFrame(data)

# Display the DataFrame (optional)
print("Original Player Stats:")
print(df)

# Step 2: Apply weights to each stat, clearly associating weights with each stat
weights_dict = {
    'FGM': 0.08,    # Field Goals Made
    'FG%': 0.07,    # Field Goal Percentage
    'PPG': 0.10,    # Points Per Game
    'APG': 0.08,    # Assists Per Game
    'MPG': 0.05,    # Minutes Per Game
    '3FGM': 0.06,   # Three-Point Field Goals Made
    '3FG%': 0.06,   # Three-Point Field Goal Percentage
    '2FGM': 0.06,   # Two-Point Field Goals Made
    '2FG%': 0.06,   # Two-Point Field Goal Percentage
    'TRB': 0.07,    # Total Rebounds
    'OREB': 0.05,   # Offensive Rebounds
    'DREB': 0.05,   # Defensive Rebounds
    'eFG%': 0.06,   # Effective Field Goal Percentage
    'FT%': 0.05,    # Free Throw Percentage
    'FTM': 0.05,    # Free Throws Made
    'TOV': 0.04,    # Turnovers
    'SPG': 0.05,    # Steals Per Game
    'BPG': 0.04     # Blocks Per Game
}

# Step 3: Ensure the weights are in the same order as the columns in the DataFrame
stats_columns = [col for col in df.columns if col != 'Player']
weights = np.array([weights_dict[stat] for stat in stats_columns])

# Step 4: Apply the weights to the raw stats
weighted_stats = df[stats_columns].values * weights

# Print weighted stats to inspect values (optional)
print("\nWeighted Stats:")
for i, player in enumerate(df['Player']):
    print(f"{player}: {weighted_stats[i]}")

# Step 5: Compute cosine similarity and convert to percentages
similarity_matrix = cosine_similarity(weighted_stats) * 100  # Convert to percentage

# Create a similarity DataFrame for easier visualization
similarity_df = pd.DataFrame(similarity_matrix, index=df['Player'], columns=df['Player'])

# Step 6: Create a DataFrame to show Player 1's similarity with Players 2 and 3
player_1_similarities = similarity_df.loc['Player 1'].drop('Player 1')
player_1_comparison_df = player_1_similarities[['Player 2', 'Player 3']].to_frame()
player_1_comparison_df.columns = ['Similarity with Player 1']

print("\nPlayer Similarity:")
print(player_1_comparison_df)
