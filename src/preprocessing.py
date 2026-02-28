# ============================================================
# STEP 3: DATA CLEANING AND PREPROCESSING
# Project: AI-Based Smart Trek Recommendation System for Nepal
# ============================================================

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================================================
# SECTION 3.0 — LOAD THE RAW DATASET
# ============================================================

df = pd.read_csv(r"E:\AI Project for third sem\Trek Data.csv")

print("=" * 60)
print("RAW DATASET LOADED")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("=" * 60)


# ============================================================
# SECTION 3.1 — INITIAL DATASET INSPECTION
# WHY: Before cleaning, we must understand the full scope
#      of problems in the data.
# ============================================================

print("\n--- df.head() ---")
print(df.head())

print("\n--- df.info() ---")
print(df.info())

print("\n--- df.describe() ---")
print(df.describe(include='all'))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(f"Total duplicates: {df.duplicated().sum()}")


# ============================================================
# SECTION 3.2 — DROP UNNECESSARY COLUMNS
# WHY: 'Unnamed: 0' is just a duplicate row index added by
#      pandas when saving. 'Contact or Book your Trip' contains
#      URLs that have no predictive value for recommendations.
#      Keeping irrelevant features adds noise to the model.
# ============================================================

print("\n[STEP 3.2] Dropping unnecessary columns...")

df.drop(columns=['Unnamed: 0', 'Contact or Book your Trip'], inplace=True)

print(f"Columns after dropping: {list(df.columns)}")


# ============================================================
# SECTION 3.3 — RENAME COLUMNS
# WHY: Consistent, lowercase, underscore-separated column names
#      follow Python/pandas best practices and prevent errors.
#      'Accomodation' is also misspelled in the original data.
# ============================================================

print("\n[STEP 3.3] Renaming columns...")

df.rename(columns={
    'Trek'            : 'trek_name',
    'Cost'            : 'cost_usd',
    'Time'            : 'duration_days',
    'Trip Grade'      : 'difficulty_level',
    'Max Altitude'    : 'max_altitude_m',
    'Accomodation'    : 'accommodation',
    'Best Travel Time': 'best_season'
}, inplace=True)

print(f"Renamed columns: {list(df.columns)}")


# ============================================================
# SECTION 3.4 — FIX AND CONVERT 'cost_usd' COLUMN
# WHY: Raw values contain '\n', '$', commas, extra spaces,
#      and the word 'USD'. We must extract only the numeric
#      value so it can be used in mathematical calculations.
# Example raw: '\n$1,420     USD'  →  1420.0
# ============================================================

print("\n[STEP 3.4] Cleaning 'cost_usd' column...")

def clean_cost(value):
    """
    Extracts numeric cost value from messy string.
    Steps: remove newline → remove $ → remove USD →
           remove commas → strip spaces → convert to float
    """
    if pd.isnull(value):
        return np.nan
    value = str(value)
    value = value.replace('\n', '')   # remove newline characters
    value = value.replace('$', '')    # remove dollar sign
    value = value.replace('USD', '')  # remove currency label
    value = value.replace(',', '')    # remove thousand separators
    value = value.strip()             # remove leading/trailing spaces
    try:
        return float(value)
    except ValueError:
        return np.nan  # return NaN if conversion fails

df['cost_usd'] = df['cost_usd'].apply(clean_cost)

print("Sample cleaned costs:")
print(df['cost_usd'].head(10).tolist())
print(f"Nulls after cleaning: {df['cost_usd'].isnull().sum()}")


# ============================================================
# SECTION 3.5 — FIX AND CONVERT 'duration_days' COLUMN
# WHY: Raw values like ' 16 Days' contain the word 'Days'
#      and leading spaces. We need a plain integer to compute
#      similarity with user's available days.
# Example raw: ' 16 Days'  →  16
# ============================================================

print("\n[STEP 3.5] Cleaning 'duration_days' column...")

def clean_duration(value):
    """
    Extracts the number of days from strings like ' 16 Days'.
    """
    if pd.isnull(value):
        return np.nan
    value = str(value)
    value = value.lower().replace('days', '').strip()
    try:
        return int(value)
    except ValueError:
        return np.nan

df['duration_days'] = df['duration_days'].apply(clean_duration)

print("Sample cleaned durations:")
print(df['duration_days'].head(10).tolist())
print(f"Nulls after cleaning: {df['duration_days'].isnull().sum()}")


# ============================================================
# SECTION 3.6 — FIX AND CONVERT 'max_altitude_m' COLUMN
# WHY: Values like '5,360 m', '4,200 m', '2012m' all represent
#      altitude but are inconsistently formatted. We extract
#      only the numeric part for model use.
# Example raw: '4,200 m'  →  4200.0
# ============================================================

print("\n[STEP 3.6] Cleaning 'max_altitude_m' column...")

def clean_altitude(value):
    """
    Extracts numeric altitude from strings like '5,360 m' or '2012m'.
    """
    if pd.isnull(value):
        return np.nan
    value = str(value)
    value = value.lower().replace('m', '')  # remove 'm' unit
    value = value.replace(',', '')          # remove thousand separator
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return np.nan

df['max_altitude_m'] = df['max_altitude_m'].apply(clean_altitude)

print("Sample cleaned altitudes:")
print(df['max_altitude_m'].head(10).tolist())
print(f"Nulls after cleaning: {df['max_altitude_m'].isnull().sum()}")


# ============================================================
# SECTION 3.7 — STANDARDISE 'difficulty_level' COLUMN
# WHY: The raw data has 10+ inconsistent grade labels.
#      For the model to work, we need a clean ordinal scale.
#      We map all variants to: Easy / Moderate / Hard
#
# Mapping logic:
#   Light, Easy                          → Easy
#   Light+Moderate, Easy To Moderate     → Easy-Moderate (→ Moderate)
#   Moderate, Moderate+Demanding         → Moderate
#   Demanding, Strenuous, Hard           → Hard
#   Demanding+Challenging                → Hard
# ============================================================

print("\n[STEP 3.7] Standardising 'difficulty_level' column...")
print("Unique raw values:", df['difficulty_level'].unique())

def standardise_difficulty(value):
    """
    Maps inconsistent difficulty labels to three standard categories.
    """
    if pd.isnull(value):
        return 'Moderate'  # default imputation
    
    value = str(value).strip().lower()
    
    # Easy group
    if value in ['easy', 'light']:
        return 'Easy'
    
    # Easy-Moderate group → classify as Moderate (middle ground)
    elif value in ['easy to moderate', 'light+moderate', 'light + moderate',
                   'moderate-easy', 'easy-moderate']:
        return 'Moderate'
    
    # Moderate group
    elif value in ['moderate']:
        return 'Moderate'
    
    # Moderate-Hard group → classify as Hard
    elif value in ['moderate+demanding', 'moderate + demanding',
                   'moderate-hard', 'hard-moderate']:
        return 'Hard'
    
    # Hard group
    elif value in ['demanding', 'strenuous', 'hard', 'challenging',
                   'demanding+challenging', 'demanding + challenging']:
        return 'Hard'
    
    else:
        # Fallback: keyword-based detection
        if 'easy' in value and 'moderate' not in value:
            return 'Easy'
        elif 'demand' in value or 'strenuous' in value or 'challeng' in value:
            return 'Hard'
        else:
            return 'Moderate'

df['difficulty_level'] = df['difficulty_level'].apply(standardise_difficulty)

print("Unique values after standardisation:", df['difficulty_level'].unique())
print(df['difficulty_level'].value_counts())


# ============================================================
# SECTION 3.8 — STANDARDISE 'accommodation' COLUMN
# WHY: The same concept appears as 'Hotel/Guesthouse',
#      'Hotel/Guest Houses', 'Hotel/Guesthouses' etc.
#      We consolidate into clean categories for encoding.
# ============================================================

print("\n[STEP 3.8] Standardising 'accommodation' column...")
print("Unique raw values:", df['accommodation'].unique())

def standardise_accommodation(value):
    """
    Consolidates messy accommodation labels into clean categories:
    - Teahouse  (basic mountain huts)
    - Guesthouse (standard guesthouses/hotels)
    - Lodge     (lodges or luxury lodges)
    """
    if pd.isnull(value):
        return 'Guesthouse'
    
    value = str(value).strip().lower()
    
    if 'teahouse' in value or 'teahouses' in value:
        return 'Teahouse'
    elif 'lodge' in value:
        return 'Lodge'
    elif 'guest' in value or 'guesthouse' in value or 'hotel' in value:
        return 'Guesthouse'
    else:
        return 'Guesthouse'

df['accommodation'] = df['accommodation'].apply(standardise_accommodation)

print("Unique values after standardisation:", df['accommodation'].unique())
print(df['accommodation'].value_counts())


# ============================================================
# SECTION 3.9 — STANDARDISE 'best_season' COLUMN
# WHY: Values like 'March - May & Sept - Dec',
#      'March - May & Setpt - Dec' (typo!), 'Jan - May & Sept - Dec'
#      all represent seasonal windows. We map them to clean
#      season labels usable in similarity calculations.
#
# Nepal trekking seasons:
#   Spring  = March - May
#   Autumn  = Sept - Nov/Dec
#   Winter  = Dec - Feb  / Jan - Feb
#   Summer  = June - Aug (monsoon)
# ============================================================

print("\n[STEP 3.9] Standardising 'best_season' column...")
print("Unique raw values:", df['best_season'].unique())

def standardise_season(value):
    """
    Maps date-range strings to clean season labels.
    Most Nepal treks are best in Spring and/or Autumn.
    If both are mentioned, we label: 'Spring & Autumn'
    """
    if pd.isnull(value):
        return 'Spring & Autumn'
    
    value = str(value).strip().lower()
    
    # Fix known typo: 'setpt' → 'sept'
    value = value.replace('setpt', 'sept')
    
    has_spring = any(m in value for m in ['march', 'april', 'may'])
    has_autumn = any(m in value for m in ['sept', 'oct', 'nov'])
    has_winter = any(m in value for m in ['dec', 'jan', 'feb'])
    has_summer = any(m in value for m in ['june', 'july', 'aug'])
    
    if has_spring and (has_autumn or has_winter):
        return 'Spring & Autumn'
    elif has_spring:
        return 'Spring'
    elif has_autumn or has_winter:
        return 'Autumn'
    elif has_summer:
        return 'Summer'
    else:
        return 'Spring & Autumn'  # default for unrecognised

df['best_season'] = df['best_season'].apply(standardise_season)

print("Unique values after standardisation:", df['best_season'].unique())
print(df['best_season'].value_counts())


# ============================================================
# SECTION 3.10 — CLEAN 'trek_name' COLUMN
# WHY: Trim whitespace and normalise to title case for
#      consistent display in recommendations output.
# ============================================================

print("\n[STEP 3.10] Cleaning 'trek_name' column...")

df['trek_name'] = df['trek_name'].str.strip().str.title()

print("Sample trek names:")
print(df['trek_name'].head(5).tolist())


# ============================================================
# SECTION 3.11 — HANDLE MISSING VALUES
# WHY: Machine learning models cannot process NaN values.
#      We use median imputation for numeric columns (robust
#      to outliers) and mode for categorical columns.
# ============================================================

print("\n[STEP 3.11] Handling missing values...")
print("Missing before imputation:")
print(df.isnull().sum())

# Numeric columns → fill with median
for col in ['cost_usd', 'duration_days', 'max_altitude_m']:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  '{col}' NaNs filled with median: {median_val}")

# Categorical columns → fill with mode
for col in ['difficulty_level', 'accommodation', 'best_season']:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    print(f"  '{col}' NaNs filled with mode: {mode_val}")

print("\nMissing after imputation:")
print(df.isnull().sum())


# ============================================================
# SECTION 3.12 — REMOVE DUPLICATE RECORDS
# WHY: Duplicate records bias the model towards certain treks,
#      inflating their similarity scores unfairly.
# ============================================================

print("\n[STEP 3.12] Removing duplicates...")

before = len(df)
df.drop_duplicates(subset=['trek_name'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
after = len(df)

print(f"Rows before: {before} | Rows after: {after} | Removed: {before - after}")


# ============================================================
# SECTION 3.13 — ENCODE CATEGORICAL FEATURES
# WHY: Cosine similarity and most ML algorithms require
#      numeric input. We convert categorical text to numbers.
#
# Strategy:
#   - Ordinal categories (difficulty, fitness) → Label Encoding
#     because there is a meaningful order: Easy < Moderate < Hard
#   - Nominal categories (accommodation, season) → One-Hot Encoding
#     because there is no inherent order between categories
# ============================================================

print("\n[STEP 3.13] Encoding categorical features...")

# --- 3.13a: Ordinal Encoding for difficulty_level ---
# Order matters: Easy=0, Moderate=1, Hard=2
difficulty_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_map)
print("difficulty_level encoded (ordinal):")
print(df[['difficulty_level', 'difficulty_encoded']].drop_duplicates())

# --- 3.13b: Derive fitness_required from difficulty ---
# WHY: Original dataset has no fitness column.
# We logically derive it from difficulty level.
fitness_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
df['fitness_encoded'] = df['difficulty_level'].map(fitness_map)
print("\nfitness_encoded (derived from difficulty):")
print(df['fitness_encoded'].value_counts())

# --- 3.13c: One-Hot Encoding for accommodation ---
accommodation_dummies = pd.get_dummies(
    df['accommodation'], prefix='acc'
)
df = pd.concat([df, accommodation_dummies], axis=1)
print("\nAccommodation one-hot columns:", list(accommodation_dummies.columns))

# --- 3.13d: One-Hot Encoding for best_season ---
season_dummies = pd.get_dummies(
    df['best_season'], prefix='season'
)
df = pd.concat([df, season_dummies], axis=1)
print("Season one-hot columns:", list(season_dummies.columns))


# ============================================================
# SECTION 3.14 — FEATURE SELECTION
# WHY: We select only features that are relevant to the user's
#      preferences. Irrelevant features add noise and reduce
#      recommendation quality.
#
# Selected features and justification:
#   duration_days    → user inputs how many days they have
#   cost_usd         → user inputs their budget
#   max_altitude_m   → proxy for physical challenge level
#   difficulty_encoded → directly matches user fitness level
#   fitness_encoded  → reinforces difficulty matching
#   acc_*            → accommodation preference
#   season_*         → preferred season matching
# ============================================================

print("\n[STEP 3.14] Selecting features...")

feature_columns = (
    ['duration_days', 'cost_usd', 'max_altitude_m',
     'difficulty_encoded', 'fitness_encoded']
    + list(accommodation_dummies.columns)
    + list(season_dummies.columns)
)

print(f"Total features selected: {len(feature_columns)}")
print("Features:", feature_columns)


# ============================================================
# SECTION 3.15 — FEATURE NORMALISATION (MinMaxScaler)
# WHY: Features have very different scales:
#      cost_usd ranges from ~100 to ~3000
#      max_altitude_m ranges from ~1600 to ~5700
#      duration_days ranges from 1 to 21
#      Without normalisation, large-scale features dominate
#      cosine similarity calculations unfairly.
#      MinMaxScaler maps all values to range [0, 1].
# ============================================================

print("\n[STEP 3.15] Normalising features with MinMaxScaler...")

scaler = MinMaxScaler()
df_features = df[feature_columns].copy()
df_features_scaled = pd.DataFrame(
    scaler.fit_transform(df_features),
    columns=feature_columns
)

print("Before scaling (sample):")
print(df[['duration_days', 'cost_usd', 'max_altitude_m']].head(3))

print("\nAfter scaling (sample):")
print(df_features_scaled[['duration_days', 'cost_usd', 'max_altitude_m']].head(3))


# ============================================================
# SECTION 3.16 — SAVE CLEANED DATASET
# WHY: Saving the cleaned data avoids re-running preprocessing
#      every time the model is used. It also creates a clean
#      reproducible dataset for academic submission.
# ============================================================

print("\n[STEP 3.16] Saving cleaned dataset...")

# Save full cleaned dataframe (with original columns)
df.to_csv("trek_dataset_cleaned.csv", index=False)

# Save scaled feature matrix (used directly by the model)
df_features_scaled['trek_name'] = df['trek_name'].values
df_features_scaled.to_csv("trek_features_scaled.csv", index=False)

print("Saved: trek_dataset_cleaned.csv")
print("Saved: trek_features_scaled.csv")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print(f"Final dataset shape: {df.shape}")
print(f"Feature matrix shape: {df_features_scaled.shape}")
print("=" * 60)

# Final preview
print("\nCleaned Dataset Preview:")
print(df[['trek_name', 'duration_days', 'cost_usd',
          'max_altitude_m', 'difficulty_level',
          'accommodation', 'best_season']].head(10).to_string())

# NOTE FOR STUDENTS:
# If you see a ChainedAssignmentError warning with fillna,
# replace: df[col].fillna(value, inplace=True)
# with:    df[col] = df[col].fillna(value)
# This is a pandas v2.0+ best practice update.