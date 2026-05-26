"""
Run this script ONCE on your local machine before deploying to Vercel.
It trains the models and saves them to models.pkl so Vercel doesn't
have to retrain on every request (which would time out).

Usage:
    pip install pandas numpy scikit-learn xgboost openpyxl
    python serialize_models.py

Make sure wrought_alloys_final.xlsx is in the same folder as this script.
"""

import os
import re
import pickle
import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

ELEMENTS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']


def parse_val(val):
    if pd.isna(val) or val == '':
        return 0.0
    s = str(val).strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
    if len(nums) == 2:
        return sum(nums) / 2
    if len(nums) == 1:
        return nums[0]
    return 0.0


def simplify_temper(t):
    t = str(t).replace('-', '').upper().strip()
    if t.startswith('T'):
        return t[:2]
    if t.startswith('H'):
        return t[:2]
    if t.startswith(('O', 'F', 'W')):
        return t[:1]
    return t


def load_and_prepare(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()

    if 'Series' in df.columns:
        df = df[~df['Series'].str.contains('2xxx|7xxx|2000|7000', case=False, na=False)]

    feature_candidates = [
        'YS (MPa)', 'EC Volume (% IACS)', 'TC (W/m-K)', 'TE Coeff', 'Fatigue Strength (MPa)'
    ]
    for col in ELEMENTS + feature_candidates:
        if col in df.columns:
            df[col] = df[col].apply(parse_val).fillna(0.0)

    if 'Temper' in df.columns:
        df = df.dropna(subset=['Temper'])
        df['Temper'] = df['Temper'].astype(str).str.strip().str.upper()
        df = df[df['Temper'] != '']
        df['Base_Temper'] = df['Temper'].apply(simplify_temper)
    else:
        df['Base_Temper'] = 'Unknown'

    cluster_features = ['YS (MPa)', 'EC Volume (% IACS)', 'TC (W/m-K)', 'TE Coeff', 'Fatigue Strength (MPa)']
    df_cluster = df.copy()
    for col in cluster_features:
        if col not in df_cluster.columns:
            df_cluster[col] = 0.0

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_cluster[cluster_features])

    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster_ID'] = km.fit_predict(scaled)
    df['Cluster_ID'] = df['Cluster_ID'].apply(lambda x: f"Cluster_{x}")

    return df, km, scaler


def train_models(df):
    feature_sets = {
        'Set 1 (YS & EC)': ['YS (MPa)', 'EC Volume (% IACS)'],
        'Set 2 (TC & TE)': ['TC (W/m-K)', 'TE Coeff'],
        'Set 3 (YS & Fatigue)': ['YS (MPa)', 'Fatigue Strength (MPa)'],
    }

    rf_params  = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'class_weight': ['balanced']}
    gb_params  = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}

    results = {}

    for set_name, prop_cols in feature_sets.items():
        print(f"  Training {set_name}...")
        df_valid = df.copy()
        for col in prop_cols:
            if col in df_valid.columns:
                df_valid = df_valid[df_valid[col] > 0]

        if len(df_valid) < 50:
            print(f"    Skipped — only {len(df_valid)} rows with non-zero values")
            continue

        feats = ELEMENTS + prop_cols
        X = df_valid[feats]
        y_raw = df_valid['Cluster_ID'].astype(str)

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        rf  = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params,
                                  n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        gb  = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gb_params,
                                  n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        xbc = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False,
                                                     eval_metric='mlogloss'), xgb_params,
                                  n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)

        rf.fit(X_tr, y_tr);  acc_rf  = accuracy_score(y_te, rf.best_estimator_.predict(X_te))
        gb.fit(X_tr, y_tr);  acc_gb  = accuracy_score(y_te, gb.best_estimator_.predict(X_te))
        xbc.fit(X_tr, y_tr); acc_xbc = accuracy_score(y_te, xbc.best_estimator_.predict(X_te))

        best_acc = max(acc_rf, acc_gb, acc_xbc)
        if best_acc == acc_rf:
            winner = rf.best_estimator_; algo = "RandomForest"; acc = acc_rf
        elif best_acc == acc_gb:
            winner = gb.best_estimator_; algo = "GradientBoosting"; acc = acc_gb
        else:
            winner = xbc.best_estimator_; algo = "XGBoost"; acc = acc_xbc

        print(f"    Best: {algo} — {round(acc*100,2)}% accuracy")

        results[set_name] = {
            'model': winner, 'encoder': le, 'features': prop_cols,
            'algorithm': algo, 'accuracy': round(acc * 100, 2)
        }

    return results


if __name__ == '__main__':
    DATA_FILE = 'wrought_alloys_final.xlsx'

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: '{DATA_FILE}' not found in current directory.")
        print("Place wrought_alloys_final.xlsx next to this script and re-run.")
        exit(1)

    print("Loading and preparing data...")
    base_df, kmeans_model, scaler_model = load_and_prepare(DATA_FILE)
    print(f"  Loaded {len(base_df)} rows.")

    print("Training models (this takes 1–3 minutes)...")
    models = train_models(base_df)

    print("Saving models.pkl ...")
    with open('models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'base_df': base_df,
            'kmeans_model': kmeans_model,
            'scaler_model': scaler_model,
        }, f)

    size_mb = os.path.getsize('models.pkl') / 1024 / 1024
    print(f"Done! models.pkl saved ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  1. Copy models.pkl into the repo root (next to vercel.json)")
    print("  2. git add models.pkl")
    print("  3. vercel --prod")