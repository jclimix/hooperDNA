import numpy as np
import logging
from loguru import logger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleAlgorithm:
    def match(self, college_player_df, nba_players_df, weights_df):
        try:
            if len(college_player_df) != 1:
                raise ValueError("college_player_df should contain only one row.")
            
            logger.info("Starting DNA match calculation using SimpleAlgorithm...")

            position_mapping = {
                "G": ["PG", "SG"],
                "F": ["SF", "PF"],
                "C": ["C"]
            }
            
            college_position = college_player_df.iloc[0]["Pos"]
            if college_position not in position_mapping:
                raise ValueError("Invalid college position")

            valid_nba_positions = position_mapping[college_position]
            nba_filtered_df = nba_players_df[nba_players_df["Pos"].isin(valid_nba_positions)].copy()

            logger.info(f"Filtered NBA DataFrame to {len(nba_filtered_df)} rows based on position.")
            
            stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
            filtered_weights = weights_df.loc[stat_columns].values.flatten()

            if np.linalg.norm(filtered_weights) != 0:
                filtered_weights /= np.linalg.norm(filtered_weights)

            college_stats = college_player_df[stat_columns].values.flatten().astype(float)
            dna_matches = []

            for idx, nba_row in nba_filtered_df.iterrows():
                nba_stats = nba_row[stat_columns].values.flatten().astype(float)
                valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)

                if not valid_indices.any():
                    dna_matches.append(np.nan)
                    continue

                college_stats_valid = college_stats[valid_indices]
                nba_stats_valid = nba_stats[valid_indices]
                weights_valid = filtered_weights[valid_indices]
                weighted_college_stats = college_stats_valid * weights_valid
                weighted_nba_stats = nba_stats_valid * weights_valid

                dot_product = np.dot(weighted_college_stats, weighted_nba_stats)
                norm_college = np.linalg.norm(weighted_college_stats)
                norm_nba = np.linalg.norm(weighted_nba_stats)

                if norm_college == 0 or norm_nba == 0:
                    dna_matches.append(np.nan)
                    continue

                cosine_similarity = dot_product / (norm_college * norm_nba)
                absolute_differences = np.abs(weighted_college_stats - weighted_nba_stats)
                penalty_factor = min(1, np.mean(absolute_differences) / np.max(absolute_differences)) if np.max(absolute_differences) != 0 else 0
                adjusted_similarity_score = cosine_similarity * (1 - penalty_factor)
                similarity_score = round(adjusted_similarity_score * 100, 1)
                similarity_score = max(0, min(100, similarity_score))
                
                dna_matches.append(similarity_score)

            nba_filtered_df["DNA Match"] = dna_matches
            logger.info("Simple DNA match calculation completed.")
            return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in SimpleAlgorithm.match: {e}")
            raise

class LegacyAlgorithm:
    def match(self, college_player_df, nba_players_df, weights_df):
        if len(college_player_df) != 1:
            raise ValueError("college_player_df should contain only one row.")
        
        position_mapping = {
            "G": ["PG", "SG"],
            "F": ["SF", "PF"],
            "C": ["C"]
        }
        
        college_position = college_player_df.iloc[0]["Pos"]
        if college_position not in position_mapping:
            raise ValueError("Invalid college position")

        valid_nba_positions = position_mapping[college_position]
        nba_filtered_df = nba_players_df[nba_players_df["Pos"].isin(valid_nba_positions)].copy()
        
        stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
        filtered_weights = weights_df.loc[stat_columns].values.flatten()
        college_stats = college_player_df[stat_columns].values.flatten().astype(float)
        dna_matches = []

        for _, nba_row in nba_filtered_df.iterrows():
            nba_stats = nba_row[stat_columns].values.flatten().astype(float)
            valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)

            if not valid_indices.any():
                continue

            college_stats_valid = college_stats[valid_indices]
            nba_stats_valid = nba_stats[valid_indices]
            weights_valid = filtered_weights[valid_indices]
            weighted_diff = (college_stats_valid - nba_stats_valid) * weights_valid
            distance = np.linalg.norm(weighted_diff)
            max_distance = np.sqrt(len(stat_columns)) * np.max([np.ptp(college_stats_valid * weights_valid), np.ptp(nba_stats_valid * weights_valid)]) or 1
            similarity_score = round(100 * (1 - (distance / max_distance)), 1)
            dna_matches.append(similarity_score)

        nba_filtered_df["DNA Match"] = dna_matches
        return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

def run_algorithm(algo_type, college_player_df, nba_players_df, weights_df):
    if algo_type == 'simple':
        algo = SimpleAlgorithm()
    elif algo_type == 'legacy':
        algo = LegacyAlgorithm()
    else:
        raise ValueError(f"Algorithm type '{algo_type}' not recognized")
    
    return algo.match(college_player_df, nba_players_df, weights_df)
