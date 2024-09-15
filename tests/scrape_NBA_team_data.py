from basketball_reference_scraper.teams import get_roster
from basketball_reference_scraper.players import get_stats

# Fetch the GSW roster for 2019
df = get_roster('GSW', 2019)
i = 0

# Loop through the first 4 players of the roster
while i < 1:
    player_name = df.iloc[i]['PLAYER']  # Accessing the player name from the DataFrame
    stats = get_stats(player_name, stat_type='PER_GAME', playoffs=False, career=False)
    print(stats)
    i += 1
