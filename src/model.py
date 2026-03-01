# ============================================================
# model.py
# Project: AI-Based Smart Trek Recommendation System for Nepal
# Purpose: Recommendation engine — imports preprocessing,
#          does NOT duplicate cleaning logic.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Import preprocessing from its own dedicated file
from preprocessing import preprocess


# ============================================================
# MODULE 1: DATA LOADING
# ============================================================

def load_data(raw_filepath):
    """
    Loads the raw CSV and calls preprocess() from preprocessing.py.
    No cleaning logic lives here — single responsibility principle.

    Parameters:
        raw_filepath (str): Path to Trek_Data.csv

    Returns:
        df, df_scaled, feature_cols, scaler
    """
    df_raw = pd.read_csv(raw_filepath)
    df, df_scaled, feature_cols, scaler = preprocess(df_raw)
    print(f"[INFO] Data loaded and cleaned: {len(df)} treks | {len(feature_cols)} features")
    return df, df_scaled, feature_cols, scaler


# ============================================================
# MODULE 2: BUILD USER QUERY VECTOR
# ============================================================

def build_user_vector(user_input, feature_cols, scaler):
    """
    Converts raw user preferences into a normalised feature vector.

    Parameters:
        user_input   (dict)        : User preferences
        feature_cols (list)        : Feature column names
        scaler       (MinMaxScaler): Fitted scaler from preprocessing

    Returns:
        user_scaled (np.array): Normalised 1D feature vector
    """
    difficulty_map   = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
    altitude_proxy   = {0: 3000, 1: 4200, 2: 5300}
    diff_enc         = difficulty_map.get(user_input['difficulty_level'], 1)

    # Start with zeros for all feature columns
    user_row = {col: 0.0 for col in feature_cols}

    # Fill numeric features
    user_row['duration_days']      = user_input['duration_days']
    user_row['cost_usd']           = user_input['cost_usd']
    user_row['max_altitude_m']     = altitude_proxy[diff_enc]
    user_row['difficulty_encoded'] = diff_enc
    user_row['fitness_encoded']    = diff_enc

    # Fill one-hot accommodation
    acc_col = f"acc_{user_input['accommodation']}"
    if acc_col in user_row:
        user_row[acc_col] = 1.0

    # Fill one-hot season
    season_col = f"season_{user_input['best_season']}"
    if season_col in user_row:
        user_row[season_col] = 1.0

    # Scale using the SAME scaler fitted on the trek data
    user_df     = pd.DataFrame([user_row])[feature_cols]
    user_scaled = scaler.transform(user_df)
    return user_scaled


# ============================================================
# MODULE 3: COSINE SIMILARITY CALCULATION
# ============================================================

def calculate_similarity(user_vector, df_scaled, feature_cols):
    """
    Computes cosine similarity between user vector and all trek vectors.

    Returns:
        similarity_scores (np.array): One score per trek
    """
    trek_matrix       = df_scaled[feature_cols].values
    similarity_scores = cosine_similarity(user_vector, trek_matrix)
    return similarity_scores.flatten()


# ============================================================
# MODULE 4: RANKING LOGIC
# ============================================================

def rank_treks(df, similarity_scores, top_n=3):
    """
    Ranks treks by similarity score and returns top N results.

    Returns:
        results (DataFrame): Top N treks with scores and details
    """
    results = df[['trek_name', 'duration_days', 'cost_usd',
                  'max_altitude_m', 'difficulty_level',
                  'accommodation', 'best_season']].copy()

    results['similarity_score'] = similarity_scores
    results = results.sort_values('similarity_score', ascending=False)
    results = results.head(top_n).reset_index(drop=True)
    results.insert(0, 'rank', range(1, top_n + 1))
    return results


# ============================================================
# MODULE 5: RECOMMENDATION OUTPUT
# ============================================================

def display_recommendations(results, user_input):
    """
    Prints the top N recommendations in a clean readable format.
    """
    print("\n" + "=" * 60)
    print("  TOP TREK RECOMMENDATIONS FOR YOUR PROFILE")
    print("=" * 60)
    print(f"  Duration     : {user_input['duration_days']} days")
    print(f"  Budget       : ${user_input['cost_usd']:,} USD")
    print(f"  Difficulty   : {user_input['difficulty_level']}")
    print(f"  Accommodation: {user_input['accommodation']}")
    print(f"  Season       : {user_input['best_season']}")
    print("=" * 60)

    for _, row in results.iterrows():
        print(f"\n   RANK #{int(row['rank'])}  —  {row['trek_name']}")
        print(f"  ─────────────────────────────────────────")
        print(f"   Match Score    : {row['similarity_score']:.2%}")
        print(f"   Duration      : {int(row['duration_days'])} days")
        print(f"   Cost          : ${row['cost_usd']:,.0f} USD")
        print(f"    Max Altitude  : {row['max_altitude_m']:,.0f} m")
        print(f"   Difficulty    : {row['difficulty_level']}")
        print(f"   Accommodation : {row['accommodation']}")
        print(f"   Best Season   : {row['best_season']}")

    print("\n" + "=" * 60)
    print("  ℹ  Scores closer to 100% = stronger match")
    print("=" * 60 + "\n")


# ============================================================
# MODULE 6: MAIN RECOMMEND FUNCTION (single entry point)
# ============================================================

def recommend(user_input, raw_filepath, top_n=3):
    """
    Full recommendation pipeline.

    Parameters:
        user_input   (dict): User preferences
        raw_filepath (str) : Path to raw Trek_Data.csv
        top_n        (int) : Number of recommendations to return

    Returns:
        results (DataFrame): Top N recommended treks
    """
    df, df_scaled, feature_cols, scaler = load_data(raw_filepath)
    user_vector       = build_user_vector(user_input, feature_cols, scaler)
    similarity_scores = calculate_similarity(user_vector, df_scaled, feature_cols)
    results           = rank_treks(df, similarity_scores, top_n)
    display_recommendations(results, user_input)
    return results


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    test_user = {
        'duration_days'   : 14,
        'cost_usd'        : 1500,
        'difficulty_level': 'Moderate',
        'accommodation'   : 'Guesthouse',
        'best_season'     : 'Spring & Autumn'
    }

    DATA_PATH = r"E:\AI Project for third sem\Trek Data.csv"
    recommend(test_user, DATA_PATH, top_n=3)