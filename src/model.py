# ============================================================
# STEP 4 & 5: MODEL DESIGN AND IMPLEMENTATION
# Project: AI-Based Smart Trek Recommendation System for Nepal
# Algorithm: Content-Based Filtering using Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# MODULE 1: DATA LOADING
# Loads and returns both the cleaned dataset and feature matrix
# ============================================================

def load_data(raw_filepath):
    """
    Loads the raw Trek dataset and runs the full
    preprocessing pipeline to return a clean DataFrame
    and scaled feature matrix ready for the model.

    Parameters:
        raw_filepath (str): Path to the raw Trek_Data.csv

    Returns:
        df (DataFrame)        : Cleaned dataset with readable columns
        df_scaled (DataFrame) : Normalised feature matrix for similarity
        feature_cols (list)   : List of feature column names
        scaler (MinMaxScaler) : Fitted scaler (needed to transform user input)
    """
    # --- Load raw data ---
    df = pd.read_csv(raw_filepath)

    # --- Run preprocessing pipeline ---
    df, df_scaled, feature_cols, scaler = preprocess(df)

    print(f"[INFO] Data loaded: {len(df)} treks, {len(feature_cols)} features")
    return df, df_scaled, feature_cols, scaler


# ============================================================
# MODULE 2: PREPROCESSING INTEGRATION
# Full cleaning pipeline (from Step 3) wrapped into one function
# ============================================================

def preprocess(df):
    """
    Runs the complete preprocessing pipeline on the raw dataset.
    Returns cleaned df, scaled feature matrix, feature names, scaler.
    """

    # --- Drop unnecessary columns ---
    cols_to_drop = [c for c in ['Unnamed: 0', 'Contact or Book your Trip'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # --- Rename columns ---
    df.rename(columns={
        'Trek'            : 'trek_name',
        'Cost'            : 'cost_usd',
        'Time'            : 'duration_days',
        'Trip Grade'      : 'difficulty_level',
        'Max Altitude'    : 'max_altitude_m',
        'Accomodation'    : 'accommodation',
        'Best Travel Time': 'best_season'
    }, inplace=True)

    # --- Clean cost_usd ---
    def clean_cost(v):
        v = str(v).replace('\n','').replace('$','').replace('USD','').replace(',','').strip()
        try: return float(v)
        except: return np.nan

    # --- Clean duration_days ---
    def clean_duration(v):
        v = str(v).lower().replace('days','').strip()
        try: return int(v)
        except: return np.nan

    # --- Clean max_altitude_m ---
    def clean_altitude(v):
        v = str(v).lower().replace('m','').replace(',','').strip()
        try: return float(v)
        except: return np.nan

    # --- Standardise difficulty_level ---
    def standardise_difficulty(v):
        if pd.isnull(v): return 'Moderate'
        v = str(v).strip().lower()
        if v in ['easy', 'light']: return 'Easy'
        elif any(x in v for x in ['easy to moderate','light+moderate','light + moderate']): return 'Moderate'
        elif v == 'moderate': return 'Moderate'
        elif any(x in v for x in ['demanding','strenuous','challenging','hard']): return 'Hard'
        elif 'moderate' in v and any(x in v for x in ['demanding','hard']): return 'Hard'
        else: return 'Moderate'

    # --- Standardise accommodation ---
    def standardise_accommodation(v):
        if pd.isnull(v): return 'Guesthouse'
        v = str(v).strip().lower()
        if 'teahouse' in v or 'teahouses' in v: return 'Teahouse'
        elif 'lodge' in v: return 'Lodge'
        else: return 'Guesthouse'

    # --- Standardise best_season ---
    def standardise_season(v):
        if pd.isnull(v): return 'Spring & Autumn'
        v = str(v).strip().lower().replace('setpt','sept')
        has_spring = any(m in v for m in ['march','april','may'])
        has_autumn = any(m in v for m in ['sept','oct','nov'])
        has_winter = any(m in v for m in ['dec','jan','feb'])
        if has_spring and (has_autumn or has_winter): return 'Spring & Autumn'
        elif has_spring: return 'Spring'
        elif has_autumn or has_winter: return 'Autumn'
        else: return 'Spring & Autumn'

    # Apply all cleaning functions
    df['cost_usd']        = df['cost_usd'].apply(clean_cost)
    df['duration_days']   = df['duration_days'].apply(clean_duration)
    df['max_altitude_m']  = df['max_altitude_m'].apply(clean_altitude)
    df['difficulty_level']= df['difficulty_level'].apply(standardise_difficulty)
    df['accommodation']   = df['accommodation'].apply(standardise_accommodation)
    df['best_season']     = df['best_season'].apply(standardise_season)
    df['trek_name']       = df['trek_name'].str.strip().str.title()

    # Impute missing numeric values with median
    for col in ['cost_usd', 'duration_days', 'max_altitude_m']:
        df[col] = df[col].fillna(df[col].median())

    # Remove duplicates
    df.drop_duplicates(subset=['trek_name'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Encode categorical features ---
    difficulty_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
    df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_map)
    df['fitness_encoded']    = df['difficulty_encoded'].copy()

    acc_dummies    = pd.get_dummies(df['accommodation'], prefix='acc')
    season_dummies = pd.get_dummies(df['best_season'],   prefix='season')
    df = pd.concat([df, acc_dummies, season_dummies], axis=1)

    # --- Feature selection ---
    feature_cols = (
        ['duration_days', 'cost_usd', 'max_altitude_m',
         'difficulty_encoded', 'fitness_encoded']
        + list(acc_dummies.columns)
        + list(season_dummies.columns)
    )

    # --- Normalise with MinMaxScaler ---
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols
    )
    df_scaled['trek_name'] = df['trek_name'].values

    return df, df_scaled, feature_cols, scaler


# ============================================================
# MODULE 3: BUILD USER QUERY VECTOR
# Converts raw user inputs into a normalised feature vector
# that can be compared against trek vectors using cosine similarity
# ============================================================

def build_user_vector(user_input, df, df_scaled, feature_cols, scaler):
    """
    Converts the user's raw preferences into a scaled feature vector.

    Parameters:
        user_input (dict): Raw user preferences, e.g.
            {
                'duration_days'  : 10,
                'cost_usd'       : 1200,
                'difficulty_level: 'Moderate',
                'accommodation'  : 'Teahouse',
                'best_season'    : 'Spring & Autumn'
            }
        df           : Cleaned DataFrame (to know which dummy columns exist)
        df_scaled    : Scaled feature matrix (to get column structure)
        feature_cols : List of feature column names
        scaler       : Fitted MinMaxScaler from preprocessing

    Returns:
        user_vector (np.array): Normalised 1D feature vector
    """

    # --- Map difficulty to numeric ---
    difficulty_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
    difficulty_encoded = difficulty_map.get(user_input['difficulty_level'], 1)
    fitness_encoded    = difficulty_encoded  # same scale

    # --- Start with zeros for all feature columns ---
    user_row = {col: 0.0 for col in feature_cols}

    # --- Fill in numeric values (unscaled — scaler will normalise) ---
    user_row['duration_days']    = user_input['duration_days']
    user_row['cost_usd']         = user_input['cost_usd']

    # For altitude: use the midpoint of the altitude range for the difficulty
    # Easy → ~3000m, Moderate → ~4200m, Hard → ~5300m
    altitude_proxy = {0: 3000, 1: 4200, 2: 5300}
    user_row['max_altitude_m']   = altitude_proxy[difficulty_encoded]

    user_row['difficulty_encoded'] = difficulty_encoded
    user_row['fitness_encoded']    = fitness_encoded

    # --- Fill one-hot accommodation ---
    acc_col = f"acc_{user_input['accommodation']}"
    if acc_col in user_row:
        user_row[acc_col] = 1.0

    # --- Fill one-hot season ---
    season_col = f"season_{user_input['best_season']}"
    if season_col in user_row:
        user_row[season_col] = 1.0

    # --- Convert to DataFrame row and scale ---
    user_df = pd.DataFrame([user_row])[feature_cols]
    user_scaled = scaler.transform(user_df)

    return user_scaled


# ============================================================
# MODULE 4: COSINE SIMILARITY CALCULATION
# Computes similarity between user vector and all trek vectors
# ============================================================

def calculate_similarity(user_vector, df_scaled, feature_cols):
    """
    Computes cosine similarity between the user vector and
    every trek's feature vector in the dataset.

    Parameters:
        user_vector  : Normalised user preference vector (1 x n)
        df_scaled    : Scaled feature matrix (m x n)
        feature_cols : Feature column names

    Returns:
        similarity_scores (np.array): Similarity score for each trek
    """
    # Extract the trek feature matrix (exclude trek_name column)
    trek_matrix = df_scaled[feature_cols].values

    # Compute cosine similarity: returns shape (1, m) — one score per trek
    similarity_scores = cosine_similarity(user_vector, trek_matrix)

    # Flatten to 1D array of m scores
    return similarity_scores.flatten()


# ============================================================
# MODULE 5: RANKING LOGIC
# Sorts treks by similarity score and returns top N
# ============================================================

def rank_treks(df, similarity_scores, top_n=3):
    """
    Ranks treks by their similarity score and returns the top N.

    Parameters:
        df               : Cleaned DataFrame with trek details
        similarity_scores: Array of cosine similarity scores
        top_n            : Number of recommendations to return (default 3)

    Returns:
        results (DataFrame): Top N treks with their details and scores
    """
    # Add similarity scores to a copy of the dataframe
    results = df[['trek_name', 'duration_days', 'cost_usd',
                  'max_altitude_m', 'difficulty_level',
                  'accommodation', 'best_season']].copy()

    results['similarity_score'] = similarity_scores

    # Sort by score descending and return top N
    results = results.sort_values('similarity_score', ascending=False)
    results = results.head(top_n).reset_index(drop=True)

    # Rank labels: 1st, 2nd, 3rd
    results.insert(0, 'rank', range(1, top_n + 1))

    return results


# ============================================================
# MODULE 6: RECOMMENDATION OUTPUT
# Formats and displays the final recommendations clearly
# ============================================================

def display_recommendations(results, user_input):
    """
    Displays the top N trek recommendations in a clean format.

    Parameters:
        results    (DataFrame): Ranked recommendations from rank_treks()
        user_input (dict)     : Original user preferences for context
    """
    print("\n" + "=" * 60)
    print("   🏔️  TOP TREK RECOMMENDATIONS FOR YOUR PROFILE")
    print("=" * 60)
    print(f"  Your Preferences:")
    print(f"  • Duration    : {user_input['duration_days']} days")
    print(f"  • Budget      : ${user_input['cost_usd']:,} USD")
    print(f"  • Difficulty  : {user_input['difficulty_level']}")
    print(f"  • Accommodation: {user_input['accommodation']}")
    print(f"  • Season      : {user_input['best_season']}")
    print("=" * 60)

    for _, row in results.iterrows():
        print(f"\n  🥇 RANK #{int(row['rank'])}  —  {row['trek_name']}")
        print(f"  ─────────────────────────────────────────")
        print(f"  ✅ Match Score   : {row['similarity_score']:.2%}")
        print(f"  📅 Duration      : {int(row['duration_days'])} days")
        print(f"  💰 Cost          : ${row['cost_usd']:,.0f} USD")
        print(f"  🏔️  Max Altitude  : {row['max_altitude_m']:,.0f} m")
        print(f"  💪 Difficulty    : {row['difficulty_level']}")
        print(f"  🏠 Accommodation : {row['accommodation']}")
        print(f"  🌸 Best Season   : {row['best_season']}")

    print("\n" + "=" * 60)
    print("  ℹ️  Scores closer to 100% indicate a stronger match.")
    print("=" * 60 + "\n")


# ============================================================
# MODULE 7: MAIN RECOMMEND FUNCTION
# Single entry point that runs the full pipeline
# ============================================================

def recommend(user_input, raw_filepath, top_n=3):
    """
    Full recommendation pipeline from raw data to output.

    Parameters:
        user_input   (dict): User preferences
        raw_filepath (str) : Path to Trek_Data.csv
        top_n        (int) : Number of recommendations

    Returns:
        results (DataFrame): Top N recommended treks
    """
    # Step 1: Load and preprocess data
    df, df_scaled, feature_cols, scaler = load_data(raw_filepath)

    # Step 2: Build user vector from preferences
    user_vector = build_user_vector(
        user_input, df, df_scaled, feature_cols, scaler
    )

    # Step 3: Calculate cosine similarity
    similarity_scores = calculate_similarity(
        user_vector, df_scaled, feature_cols
    )

    # Step 4: Rank and retrieve top N
    results = rank_treks(df, similarity_scores, top_n)

    # Step 5: Display results
    display_recommendations(results, user_input)

    return results


# ============================================================
# QUICK TEST — runs when script is executed directly
# ============================================================

if __name__ == "__main__":

    # Example user profile for testing
    test_user = {
        'duration_days'   : 14,
        'cost_usd'        : 1500,
        'difficulty_level': 'Moderate',
        'accommodation'   : 'Guesthouse',
        'best_season'     : 'Spring & Autumn'
    }

    # ⚠️ Update this path to match your actual file location
    DATA_PATH = r"E:\AI Project for third sem\Trek Data.csv"

    print("[TEST] Running recommendation for sample user profile...")
    results = recommend(test_user, DATA_PATH, top_n=3)

    print("[TEST] Raw results DataFrame:")
    print(results.to_string(index=False))