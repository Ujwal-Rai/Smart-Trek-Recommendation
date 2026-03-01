
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df):

    cols_to_drop = [c for c in ['Unnamed: 0', 'Contact or Book your Trip'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    df.rename(columns={
        'Trek'            : 'trek_name',
        'Cost'            : 'cost_usd',
        'Time'            : 'duration_days',
        'Trip Grade'      : 'difficulty_level',
        'Max Altitude'    : 'max_altitude_m',
        'Accomodation'    : 'accommodation',
        'Best Travel Time': 'best_season'
    }, inplace=True)

    def clean_cost(v):
        v = str(v).replace('\n','').replace('$','').replace('USD','').replace(',','').strip()
        try: return float(v)
        except: return np.nan

    def clean_duration(v):
        v = str(v).lower().replace('days','').strip()
        try: return int(v)
        except: return np.nan

    def clean_altitude(v):
        v = str(v).lower().replace('m','').replace(',','').strip()
        try: return float(v)
        except: return np.nan

    def standardise_difficulty(v):
        if pd.isnull(v): return 'Moderate'
        v = str(v).strip().lower()
        if v in ['easy', 'light']: return 'Easy'
        elif any(x in v for x in ['easy to moderate','light+moderate','light + moderate']): return 'Moderate'
        elif v == 'moderate': return 'Moderate'
        elif any(x in v for x in ['demanding','strenuous','challenging','hard']): return 'Hard'
        elif 'moderate' in v and any(x in v for x in ['demanding','hard']): return 'Hard'
        else: return 'Moderate'

    def standardise_accommodation(v):
        if pd.isnull(v): return 'Guesthouse'
        v = str(v).strip().lower()
        if 'teahouse' in v or 'teahouses' in v: return 'Teahouse'
        elif 'lodge' in v: return 'Lodge'
        else: return 'Guesthouse'

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

    df['cost_usd']         = df['cost_usd'].apply(clean_cost)
    df['duration_days']    = df['duration_days'].apply(clean_duration)
    df['max_altitude_m']   = df['max_altitude_m'].apply(clean_altitude)
    df['difficulty_level'] = df['difficulty_level'].apply(standardise_difficulty)
    df['accommodation']    = df['accommodation'].apply(standardise_accommodation)
    df['best_season']      = df['best_season'].apply(standardise_season)
    df['trek_name']        = df['trek_name'].str.strip().str.title()


    for col in ['cost_usd', 'duration_days', 'max_altitude_m']:
        df[col] = df[col].fillna(df[col].median())


    df.drop_duplicates(subset=['trek_name'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)


    difficulty_map = {'Easy': 0, 'Moderate': 1, 'Hard': 2}
    df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_map)
    df['fitness_encoded']    = df['difficulty_encoded'].copy()

    acc_dummies    = pd.get_dummies(df['accommodation'], prefix='acc')
    season_dummies = pd.get_dummies(df['best_season'],   prefix='season')
    df = pd.concat([df, acc_dummies, season_dummies], axis=1)

    feature_cols = (
        ['duration_days', 'cost_usd', 'max_altitude_m',
         'difficulty_encoded', 'fitness_encoded']
        + list(acc_dummies.columns)
        + list(season_dummies.columns)
    )

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols
    )
    df_scaled['trek_name'] = df['trek_name'].values

    return df, df_scaled, feature_cols, scaler