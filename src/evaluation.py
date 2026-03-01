import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from preprocessing import preprocess

def load_data(raw_filepath):
    df_raw = pd.read_csv(raw_filepath)
    df, df_scaled, feature_cols, scaler = preprocess(df_raw)
    return df, df_scaled, feature_cols, scaler

def build_user_vector(user_input, feature_cols, scaler):
    difficulty_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
    altitude_proxy = {0: 3000, 1: 4200, 2: 5300}
    diff_enc       = difficulty_map.get(user_input['difficulty_level'], 1)

    user_row = {col: 0.0 for col in feature_cols}
    user_row['duration_days']      = user_input['duration_days']
    user_row['cost_usd']           = user_input['cost_usd']
    user_row['max_altitude_m']     = altitude_proxy[diff_enc]
    user_row['difficulty_encoded'] = diff_enc
    user_row['fitness_encoded']    = diff_enc

    acc_col    = f"acc_{user_input['accommodation']}"
    season_col = f"season_{user_input['best_season']}"
    if acc_col    in user_row: user_row[acc_col]    = 1.0
    if season_col in user_row: user_row[season_col] = 1.0

    user_df     = pd.DataFrame([user_row])[feature_cols]
    user_scaled = scaler.transform(user_df)
    return user_scaled

def get_recommendations(user_input, df, df_scaled, feature_cols, scaler, top_n=3):
    user_vector = build_user_vector(user_input, feature_cols, scaler)
    trek_matrix = df_scaled[feature_cols].values
    scores      = cosine_similarity(user_vector, trek_matrix).flatten()

    results = df[['trek_name','duration_days','cost_usd',
                  'max_altitude_m','difficulty_level',
                  'accommodation','best_season']].copy()
    results['similarity_score'] = scores
    results = results.sort_values('similarity_score', ascending=False)
    results = results.head(top_n).reset_index(drop=True)
    return results

def evaluate_similarity_scores(user_input, df, df_scaled, feature_cols, scaler):
    """
    Computes and analyses cosine similarity scores across all treks
    for a given user profile.
    """
    print("\n" + "=" * 60)
    print("  METRIC 1: SIMILARITY SCORE ANALYSIS")
    print("=" * 60)

    user_vector = build_user_vector(user_input, feature_cols, scaler)
    trek_matrix = df_scaled[feature_cols].values
    scores      = cosine_similarity(user_vector, trek_matrix).flatten()

    # Full score distribution
    score_df = pd.DataFrame({
        'trek_name'       : df['trek_name'].values,
        'difficulty_level': df['difficulty_level'].values,
        'similarity_score': scores
    }).sort_values('similarity_score', ascending=False).reset_index(drop=True)

    print(f"\n  User Profile: {user_input['duration_days']} days | "
          f"${user_input['cost_usd']} | {user_input['difficulty_level']}")
    print(f"\n  Score Statistics:")
    print(f"    Highest Score : {scores.max():.4f}  ({scores.max():.2%})")
    print(f"    Lowest Score  : {scores.min():.4f}  ({scores.min():.2%})")
    print(f"    Mean Score    : {scores.mean():.4f}  ({scores.mean():.2%})")
    print(f"    Std Deviation : {scores.std():.4f}")

    print(f"\n  Top 10 Most Similar Treks:")
    print(f"  {'Rank':<5} {'Trek Name':<40} {'Difficulty':<12} {'Score'}")
    print(f"  {'-'*5} {'-'*40} {'-'*12} {'-'*8}")
    for i, row in score_df.head(10).iterrows():
        print(f"  {i+1:<5} {row['trek_name']:<40} "
              f"{row['difficulty_level']:<12} {row['similarity_score']:.2%}")

    print(f"\n  Bottom 5 Least Similar Treks:")
    for i, row in score_df.tail(5).iterrows():
        print(f"  {row['trek_name']:<40} {row['similarity_score']:.2%}")

    return scores, score_df


def evaluate_precision_at_k(user_input, df, df_scaled, feature_cols, scaler, k_values=[1, 3, 5]):
    """
    Computes Precision@K for multiple values of K.

    A trek is considered RELEVANT if its difficulty_level
    matches the user's requested difficulty_level.

    Precision@K = (Number of relevant treks in top K) / K
    """
    print("\n" + "=" * 60)
    print("  METRIC 2: PRECISION@K")
    print("=" * 60)
    print(f"\n  Relevance criterion: difficulty_level matches user input")
    print(f"  User difficulty: '{user_input['difficulty_level']}'")

    user_vector = build_user_vector(user_input, feature_cols, scaler)
    trek_matrix = df_scaled[feature_cols].values
    scores      = cosine_similarity(user_vector, trek_matrix).flatten()

    results = df[['trek_name','difficulty_level','cost_usd','duration_days']].copy()
    results['similarity_score'] = scores
    results = results.sort_values('similarity_score', ascending=False).reset_index(drop=True)

    # Mark relevance
    results['is_relevant'] = (
        results['difficulty_level'] == user_input['difficulty_level']
    ).astype(int)

    print(f"\n  {'K':<6} {'Relevant in Top K':<22} {'Precision@K'}")
    print(f"  {'-'*6} {'-'*22} {'-'*12}")

    precision_results = {}
    for k in k_values:
        top_k       = results.head(k)
        relevant    = top_k['is_relevant'].sum()
        precision   = relevant / k
        precision_results[k] = precision
        print(f"  @K={k:<3} {relevant}/{k} relevant         {precision:.2%}")

    print(f"\n  Detailed Top-{max(k_values)} Results:")
    print(f"  {'Rank':<5} {'Trek Name':<40} {'Difficulty':<12} {'Relevant':<10} {'Score'}")
    print(f"  {'-'*5} {'-'*40} {'-'*12} {'-'*10} {'-'*8}")
    for i, row in results.head(max(k_values)).iterrows():
        relevant_label = ' Yes' if row['is_relevant'] else ' No'
        print(f"  {i+1:<5} {row['trek_name']:<40} "
              f"{row['difficulty_level']:<12} {relevant_label:<10} "
              f"{row['similarity_score']:.2%}")

    return precision_results



def evaluate_coverage(df, df_scaled, feature_cols, scaler):
    """
    Tests multiple user profiles and records which treks get recommended.
    Coverage = Unique treks recommended / Total treks in dataset
    """
    print("\n" + "=" * 60)
    print("  METRIC 3: CATALOGUE COVERAGE")
    print("=" * 60)

    test_profiles = [
        {'duration_days':  4, 'cost_usd':  400, 'difficulty_level': 'Easy',     'accommodation': 'Teahouse',   'best_season': 'Spring & Autumn'},
        {'duration_days':  7, 'cost_usd':  700, 'difficulty_level': 'Easy',     'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
        {'duration_days': 10, 'cost_usd': 1000, 'difficulty_level': 'Moderate', 'accommodation': 'Teahouse',   'best_season': 'Spring & Autumn'},
        {'duration_days': 14, 'cost_usd': 1500, 'difficulty_level': 'Moderate', 'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
        {'duration_days': 14, 'cost_usd': 1800, 'difficulty_level': 'Hard',     'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
        {'duration_days': 18, 'cost_usd': 2000, 'difficulty_level': 'Hard',     'accommodation': 'Lodge',      'best_season': 'Spring & Autumn'},
        {'duration_days': 20, 'cost_usd': 2500, 'difficulty_level': 'Hard',     'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
        {'duration_days':  5, 'cost_usd':  500, 'difficulty_level': 'Easy',     'accommodation': 'Teahouse',   'best_season': 'Spring & Autumn'},
    ]

    all_recommended = set()
    total_treks     = len(df)

    for profile in test_profiles:
        results = get_recommendations(profile, df, df_scaled, feature_cols, scaler, top_n=3)
        for trek in results['trek_name'].values:
            all_recommended.add(trek)

    coverage = len(all_recommended) / total_treks

    print(f"\n  Total treks in dataset       : {total_treks}")
    print(f"  Test profiles used           : {len(test_profiles)}")
    print(f"  Unique treks recommended     : {len(all_recommended)}")
    print(f"  Coverage Score               : {coverage:.2%}")
    print(f"\n  Treks that appeared in recommendations:")
    for t in sorted(all_recommended):
        print(f"    • {t}")

    if coverage >= 0.30:
        print(f"\n   Coverage is acceptable (≥30% of catalogue reached)")
    else:
        print(f"\n  ⚠️  Coverage is low — consider increasing top_n or diversifying features")

    return coverage, all_recommended



def evaluate_diversity(user_input, df, df_scaled, feature_cols, scaler, top_n=3):
    """
    Computes diversity of the top-N recommendations.
    Diversity = 1 - mean pairwise cosine similarity between recommended treks.
    Higher = more varied recommendations.
    """
    print("\n" + "=" * 60)
    print("  METRIC 4: RECOMMENDATION DIVERSITY")
    print("=" * 60)

    results     = get_recommendations(user_input, df, df_scaled, feature_cols, scaler, top_n)
    trek_names  = results['trek_name'].tolist()


    rec_vectors = df_scaled[df_scaled['trek_name'].isin(trek_names)][feature_cols].values

    if len(rec_vectors) < 2:
        print("  Not enough recommendations to compute diversity.")
        return 0.0

    pairwise = cosine_similarity(rec_vectors)

    
    n         = len(pairwise)
    pairs     = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_sims = [pairwise[i][j] for i, j in pairs]

    avg_similarity = np.mean(pair_sims)
    diversity      = 1 - avg_similarity

    print(f"\n  Top-{top_n} Recommended Treks:")
    for name in trek_names:
        print(f"    • {name}")

    print(f"\n  Pairwise Similarity Matrix:")
    header = f"  {'':30}" + "".join(f"{i+1:>10}" for i in range(len(trek_names)))
    print(header)
    for i, name in enumerate(trek_names):
        row_label = f"  {name[:28]:30}"
        row_vals  = "".join(f"{pairwise[i][j]:>10.4f}" for j in range(len(trek_names)))
        print(row_label + row_vals)

    print(f"\n  Average Pairwise Similarity : {avg_similarity:.4f}")
    print(f"  Diversity Score             : {diversity:.4f}  ({diversity:.2%})")

    if diversity >= 0.15:
        print(f"   Good diversity — recommendations are varied")
    else:
        print(f"  ⚠️  Low diversity — recommendations are very similar to each other")

    return diversity



def evaluate_sensitivity(df, df_scaled, feature_cols, scaler):
    """
    Tests that the model output changes meaningfully when
    user profile inputs change. This validates that the
    recommendation engine is sensitive to user preferences.
    """
    print("\n" + "=" * 60)
    print("  METRIC 5: PROFILE SENSITIVITY TEST")
    print("=" * 60)

    profiles = {
        "Beginner (Easy, 4 days, $400)"    : {'duration_days':  4, 'cost_usd':  400, 'difficulty_level': 'Easy',     'accommodation': 'Teahouse',   'best_season': 'Spring & Autumn'},
        "Intermediate (Moderate, 10, $900)": {'duration_days': 10, 'cost_usd':  900, 'difficulty_level': 'Moderate', 'accommodation': 'Teahouse',   'best_season': 'Spring & Autumn'},
        "Expert (Hard, 18 days, $2000)"    : {'duration_days': 18, 'cost_usd': 2000, 'difficulty_level': 'Hard',     'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
        "Budget Short (Easy, 2 days, $150)": {'duration_days':  2, 'cost_usd':  150, 'difficulty_level': 'Easy',     'accommodation': 'Guesthouse', 'best_season': 'Spring & Autumn'},
    }

    all_results   = {}
    print()
    for label, profile in profiles.items():
        results = get_recommendations(profile, df, df_scaled, feature_cols, scaler, top_n=3)
        top3    = results['trek_name'].tolist()
        all_results[label] = top3
        print(f"  Profile : {label}")
        for i, t in enumerate(top3):
            score = results.loc[i, 'similarity_score']
            print(f"    Rank {i+1}: {t:<40} ({score:.2%})")
        print()

    
    result_sets  = [frozenset(v) for v in all_results.values()]
    unique_sets  = len(set(result_sets))
    total_sets   = len(result_sets)

    print(f"  Unique recommendation sets : {unique_sets} / {total_sets}")
    if unique_sets == total_sets:
        print(f"   Model is fully sensitive — every profile gets unique recommendations")
    elif unique_sets > 1:
        print(f"   Model shows partial sensitivity — most profiles get different results")
    else:
        print(f"   Model is not sensitive — all profiles return identical results")

    return all_results



def run_full_evaluation(raw_filepath):
    """
    Runs the complete evaluation suite and prints a summary.

    Parameters:
        raw_filepath (str): Path to Trek_Data.csv
    """
    print("\n" + "=" * 60)
    print("   FULL MODEL EVALUATION REPORT")
    print("   AI-Based Smart Trek Recommendation System")
    print("=" * 60)

    # Load data
    df, df_scaled, feature_cols, scaler = load_data(raw_filepath)


    test_user = {
        'duration_days'   : 14,
        'cost_usd'        : 1500,
        'difficulty_level': 'Moderate',
        'accommodation'   : 'Guesthouse',
        'best_season'     : 'Spring & Autumn'
    }


    scores, score_df                = evaluate_similarity_scores(test_user, df, df_scaled, feature_cols, scaler)
    precision_results               = evaluate_precision_at_k(test_user, df, df_scaled, feature_cols, scaler, k_values=[1, 3, 5])
    coverage, recommended_treks     = evaluate_coverage(df, df_scaled, feature_cols, scaler)
    diversity                       = evaluate_diversity(test_user, df, df_scaled, feature_cols, scaler)
    sensitivity_results             = evaluate_sensitivity(df, df_scaled, feature_cols, scaler)


    print("\n" + "=" * 60)
    print("   EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Highest Similarity Score : {scores.max():.2%}")
    print(f"  Mean Similarity Score    : {scores.mean():.2%}")
    print(f"  Precision@1              : {precision_results[1]:.2%}")
    print(f"  Precision@3              : {precision_results[3]:.2%}")
    print(f"  Precision@5              : {precision_results[5]:.2%}")
    print(f"  Catalogue Coverage       : {coverage:.2%}")
    print(f"  Diversity Score          : {diversity:.2%}")
    print("=" * 60)
    print("\n  Evaluation complete.\n")


if __name__ == "__main__":
    DATA_PATH = r"E:\AI Project for third sem\Trek Data.csv"
    run_full_evaluation(DATA_PATH)