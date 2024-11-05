import numpy as np
import os, logging, boto3, pandas as pd, io
from loguru import logger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simple_calculate_dna_match(college_player_df, nba_players_df, weights_df):
    try:
        if len(college_player_df) != 1:
            raise ValueError("college_player_df should contain only one row.")
        
        logger.info("Starting DNA match calculation...")
        
        # College to NBA position mapping
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

        # Get stat columns that are common to both DataFrames
        stat_columns = list(set(college_player_df.columns) & set(nba_filtered_df.columns) - {"Season", "Team", "Conf", "Class", "Pos", "G", "GS", "Awards"})
        logger.info(f"Using stat columns: {stat_columns}")

        filtered_weights = weights_df.loc[stat_columns].values.flatten()

        # Normalize weights to prevent scale issues
        if np.linalg.norm(filtered_weights) != 0:
            filtered_weights /= np.linalg.norm(filtered_weights)

        college_stats = college_player_df[stat_columns].values.flatten().astype(float)

        dna_matches = []
        for idx, nba_row in nba_filtered_df.iterrows():
            nba_stats = nba_row[stat_columns].values.flatten().astype(float)

            valid_indices = ~np.isnan(college_stats) & ~np.isnan(nba_stats)
            
            if not valid_indices.any():
                logger.debug(f"Skipping row {idx} due to no valid indices.")
                dna_matches.append(np.nan)  # Ensure a value is added to maintain length
                continue
            
            college_stats_valid = college_stats[valid_indices]
            nba_stats_valid = nba_stats[valid_indices]
            weights_valid = filtered_weights[valid_indices]

            # Apply weights
            weighted_college_stats = college_stats_valid * weights_valid
            weighted_nba_stats = nba_stats_valid * weights_valid

            # Compute cosine similarity
            dot_product = np.dot(weighted_college_stats, weighted_nba_stats)
            norm_college = np.linalg.norm(weighted_college_stats)
            norm_nba = np.linalg.norm(weighted_nba_stats)

            if norm_college == 0 or norm_nba == 0:
                logger.debug(f"Skipping row {idx} due to zero norm.")
                dna_matches.append(np.nan)  # Ensure a value is added to maintain length
                continue

            cosine_similarity = dot_product / (norm_college * norm_nba)

            # Apply a penalty for large stat differences
            absolute_differences = np.abs(weighted_college_stats - weighted_nba_stats)
            if np.max(absolute_differences) == 0:
                penalty_factor = 0
            else:
                penalty_factor = np.mean(absolute_differences) / np.max(absolute_differences)
            
            penalty_factor = min(1, penalty_factor)  # Ensure penalty is at most 1

            # Adjust similarity score with penalty
            adjusted_similarity_score = cosine_similarity * (1 - penalty_factor)

            # Convert to a similarity score between 0 and 100
            similarity_score = round(adjusted_similarity_score * 100, 1)

            # Ensure the similarity score is within 0-100
            similarity_score = max(0, min(100, similarity_score))
            
            dna_matches.append(similarity_score)

        if len(dna_matches) != len(nba_filtered_df):
            logger.error("Mismatch in length between DNA matches and NBA DataFrame.")
            raise ValueError("Mismatch in length between DNA matches and NBA DataFrame.")

        nba_filtered_df["DNA Match"] = dna_matches

        logger.info("DNA match calculation completed successfully.")
        return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.error(f"An error occurred in calculate_dna_match: {e}")
        raise


def legacy_calculate_dna_match(college_player_df, nba_players_df, weights_df):

    if len(college_player_df) != 1:
        raise ValueError("college_player_df should contain only one row.")

    # college to NBA position mapping
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
        
        max_distance = np.sqrt(len(stat_columns)) * np.max([np.ptp(college_stats_valid * weights_valid), np.ptp(nba_stats_valid * weights_valid)])
        if max_distance == 0:
            max_distance = 1
        similarity_score = 100 * (1 - (distance / max_distance))

        similarity_score = round(similarity_score, 1)
        
        dna_matches.append(similarity_score)

    nba_filtered_df["DNA Match"] = dna_matches

    return nba_filtered_df.sort_values(by="DNA Match", ascending=False).reset_index(drop=True)
