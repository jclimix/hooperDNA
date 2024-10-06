import duckdb
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

class Player:
    def __init__(self, player_id, csv_file_path, player_type):
        self.player_id = player_id
        self.csv_file_path = csv_file_path
        self.player_type = player_type
        self.player_name = ""
        self.player_stat_dict = self.init_player_stat_dict()
    
    def init_player_stat_dict(self):
        return {
            'MP': 0.0, 'FG': 0.0, 'FGA': 0.0, 'FG%': 0.0, '3P': 0.0, '3PA': 0.0, 
            '3P%': 0.0, 'FT': 0.0, 'FTA': 0.0, 'FT%': 0.0, 'ORB': 0.0, 'DRB': 0.0,
            'TRB': 0.0, 'AST': 0.0, 'STL': 0.0, 'BLK': 0.0, 'TOV': 0.0, 'PF': 0.0, 
            'PTS': 0.0
        }

# CollegePlayer class inherits from Player
class CollegePlayer(Player):
    def __init__(self, player_id, csv_file_path):
        super().__init__(player_id, csv_file_path, "college")
        self.college_latest_stats = None
    
    def init_college_player_info(self):
        df = pd.read_csv(self.csv_file_path)
        row = df[df['playerId'] == self.player_id]
        if not row.empty:
            self.player_name = row['playerName'].values[0]
        return self.player_name, self.player_stat_dict
    
    def scrape_college_stats(self):
        url = f'https://www.sports-reference.com/cbb/players/{self.player_id}.html'
        response = requests.get(url)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')
        div = soup.find('div', id='div_players_per_game')

        if div:
            table = div.find('table')
            if table:
                college_player_stats_df = pd.read_html(str(table))[0]
                if 'Season' in college_player_stats_df.columns:
                    career_index = college_player_stats_df[college_player_stats_df['Season'] == 'Career'].index
                    if not career_index.empty:
                        college_latest_stats_index = career_index[0] - 1
                        if college_latest_stats_index >= 0:
                            self.college_latest_stats = college_player_stats_df.iloc[college_latest_stats_index]
                            for stat in self.player_stat_dict.keys():
                                if stat in self.college_latest_stats.index:
                                    self.player_stat_dict[stat] = self.college_latest_stats[stat]
                            self.adjust_college_stats()
        return self.college_latest_stats, self.player_stat_dict

    def adjust_college_stats(self):
        # College player dict adjustments (NCAA => NBA)
        self.player_stat_dict["MP"] *= 1.17  # 40 vs 48 total min
        self.player_stat_dict["PTS"] *= 1.15  # Skew scoring for better offensive player matches

# NBAMatchFinder class handles NBA comparison
class NBAMatchFinder:
    def __init__(self, college_player, selected_profile):
        self.college_player = college_player
        self.selected_profile = selected_profile
        self.weights = self.init_weight_profiles()

    def init_weight_profiles(self):
        weight_profiles = {
            'offense': {
                'MP': 6.0, 'FG': 7.0, 'FGA': 5.0, 'FG%': 6.0, '3P': 9.0, '3PA': 5.0,
                '3P%': 8.0, 'FT': 4.0, 'FTA': 3.0, 'FT%': 7.0, 'ORB': 5.0, 'DRB': 2.0, 
                'TRB': 4.0, 'AST': 7.0, 'STL': 4.0, 'BLK': 4.0, 'TOV': 3.0, 'PF': 2.0,
                'PTS': 8.0
            },
            'defense': {
                'MP': 6.0, 'FG': 4.0, 'FGA': 3.0, 'FG%': 5.0, '3P': 4.0, '3PA': 3.0,
                '3P%': 4.0, 'FT': 4.0, 'FTA': 3.0, 'FT%': 4.0, 'ORB': 7.0, 'DRB': 8.0, 
                'TRB': 8.0, 'AST': 5.0, 'STL': 9.0, 'BLK': 9.0, 'TOV': 2.0, 'PF': 6.0,
                'PTS': 4.0
            },
            'balanced': {
                'MP': 7.0, 'FG': 6.0, 'FGA': 6.0, 'FG%': 6.0, '3P': 6.0, '3PA': 6.0, 
                '3P%': 7.0, 'FT': 6.0, 'FTA': 6.0, 'FT%': 6.0, 'ORB': 6.0, 'DRB': 6.0, 
                'TRB': 6.0, 'AST': 6.0, 'STL': 6.0, 'BLK': 6.0, 'TOV': 4.0, 'PF': 5.0, 
                'PTS': 6.0
            }
        }

        raw_weights = weight_profiles[self.selected_profile]
        total_weight = sum(raw_weights.values())
        return {stat: value / total_weight for stat, value in raw_weights.items()}

    # replace NaN and Inf values
    def clean_stat_dict(self, stat_dict):
        return {k: (0 if (v is None or np.isnan(v) or np.isinf(v)) else v) for k, v in stat_dict.items()}
    
    def clean_row(self, row):
        return row.apply(lambda x: 0 if (pd.isna(x) or np.isinf(x)) else x)

    def find_nba_matches(self):
        
        # Clean the college player stats before weighting
        cleaned_college_stats = self.clean_stat_dict(self.college_player.player_stat_dict)
        college_stats = np.array([cleaned_college_stats[stat] * self.weights[stat] for stat in cleaned_college_stats.keys()]).reshape(1, -1)
        results = []
        dir = './sample_DB/nba_raw_data'

        for year in range(2015, 2025):
            csv_file = os.path.join(dir, f'{year}_NBAPlayerStats_HprDNA_raw.csv')

            if os.path.exists(csv_file):
                query = f"SELECT * FROM '{csv_file}'"
                season_data = duckdb.query(query).df()
                stat_columns = list(cleaned_college_stats.keys())

                if all(col in season_data.columns for col in stat_columns):
                    filtered_season_data = season_data[season_data['Pos'].isin(self.get_position_filter())].reset_index(drop=True)
                    
                    # Clean NBA data row-wise before applying weights
                    weighted_nba_stats = filtered_season_data[stat_columns].apply(lambda row: self.clean_row(row) * np.array([self.weights[stat] for stat in stat_columns]), axis=1)
                    distances = weighted_nba_stats.apply(lambda row: euclidean(row, college_stats.flatten()), axis=1)
                    min_dist_index = np.argmin(distances)

                    if min_dist_index in filtered_season_data.index:
                        distance_percentage = (1 / (1 + distances.iloc[min_dist_index])) * 100
                        most_similar_player_df = pd.DataFrame([filtered_season_data.iloc[min_dist_index]], columns=filtered_season_data.columns)
                        most_similar_player_df['Similarity (%)'] = f'{distance_percentage:.2f}%'
                        results.append(most_similar_player_df)

        if results:
            nba_dna_matches = pd.concat(results, ignore_index=True)
            return nba_dna_matches.sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)
        else:
            print("No matching NBA data found.")
            return pd.DataFrame()

    def get_position_filter(self):
        college_player_pos = self.college_player.college_latest_stats.get('Pos', 'UNKWN')
        pos_mapping = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C']}
        return pos_mapping.get(college_player_pos, [])

def main():

    college_player_id = 'zach-edey-1'

    selected_profile = 'offense'

    csv_file_path = './sample_DB/college_data/college_basketball_players.csv'

    college_player = CollegePlayer(college_player_id, csv_file_path)
    college_player_name, college_player_stat_dict = college_player.init_college_player_info()
    college_latest_stats, college_player_stat_dict = college_player.scrape_college_stats()

    if college_latest_stats is not None:
        nba_match_finder = NBAMatchFinder(college_player, selected_profile)
        nba_dna_matches = nba_match_finder.find_nba_matches()

        if not nba_dna_matches.empty:
            print(f"\n{college_player_name}'s NBA Player Matches (In Last Decade):")
            print(nba_dna_matches)
        else:
            print("No NBA matches found.")
    else:
        print(f"Failed to scrape stats for {college_player_id}.")

if __name__ == "__main__":
    main()
