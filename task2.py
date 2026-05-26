#part 4

import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING, PREPROCESSING & K-MEANS CLUSTERING
# =============================================================================
def load_and_preprocess_classification(file_path):
    print(f"[{'='*20} LOADING & CLUSTERING DATA FOR PAIR MODELS {'='*20}]")
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()

    if 'Series' in df.columns:
        df = df[~df['Series'].str.contains('2xxx|7xxx|2000|7000', case=False, na=False)]

    elements = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']

    def parse_val(val):
        if pd.isna(val) or val == '': return 0.0
        s = str(val).strip()
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
        if len(nums) == 2: return sum(nums)/2
        if len(nums) == 1: return nums[0]
        return 0.0

    feature_candidates = ['YS (MPa)', 'EC Volume (% IACS)', 'TC (W/m-K)', 'TE Coeff', 'Fatigue Strength (MPa)']
    for col in elements + feature_candidates:
        if col in df.columns:
            df[col] = df[col].apply(parse_val).fillna(0.0)

    # Base Temper Grouping (For Translation Tracking)
    if 'Temper' in df.columns:
        df = df.dropna(subset=['Temper'])
        df['Temper'] = df['Temper'].astype(str).str.strip().str.upper()
        df = df[df['Temper'] != '']

        def simplify_temper(t):
            t = t.replace('-', '')
            if t.startswith('T'): return t[:2]
            elif t.startswith('H'): return t[:2]
            elif t.startswith('O') or t.startswith('F') or t.startswith('W'): return t[:1]
            return t
        df['Base_Temper'] = df['Temper'].apply(simplify_temper)
    else:
        df['Base_Temper'] = 'Unknown'

    # =========================================================================
    # K-MEANS CLUSTERING
    # =========================================================================
    print(" > Applying K-Means Clustering to group alloys mathematically...")

    # We cluster based on the core properties we care about in this phase
    cluster_features = ['YS (MPa)', 'EC Volume (% IACS)', 'TC (W/m-K)', 'TE Coeff', 'Fatigue Strength (MPa)']
    df_cluster = df.copy()

    for col in cluster_features:
        if col not in df_cluster.columns:
            df_cluster[col] = 0.0

    # Scale data so large numbers (YS) don't crush small numbers (TE)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster[cluster_features])

    # Create 5 distinct "Alloy Families"
    NUM_CLUSTERS = 5
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(scaled_features)
    df['Cluster_ID'] = df['Cluster_ID'].apply(lambda x: f"Cluster_{x}")

    print(f" > Successfully created {NUM_CLUSTERS} mathematical alloy clusters.")
    print(f" > Data Loaded. Total Valid Rows: {len(df)}")

    return df, elements, kmeans, scaler

# =============================================================================
# 2. TUNED CLASSIFICATION TRAINING ENGINE
# =============================================================================
def train_classification_models_tuned(df, elements):
    print(f"\n[{'='*20} TRAINING PAIR MODELS ON CLUSTERS {'='*20}]")

    # The 3 sets of combined property features
    feature_sets = {
        'Set 1 (YS & EC)': ['YS (MPa)', 'EC Volume (% IACS)'],
        'Set 2 (TC & TE)': ['TC (W/m-K)', 'TE Coeff'],
        'Set 3 (YS & Fatigue)': ['YS (MPa)', 'Fatigue Strength (MPa)']
    }

    rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'class_weight': ['balanced']}
    gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}

    results = {}

    for set_name, prop_cols in feature_sets.items():
        print(f"\n--- Tuning {set_name} ---")
        print(f" > Input: 12 Elements + {prop_cols[0]} + {prop_cols[1]}")

        df_valid = df.copy()
        for col in prop_cols:
            if col in df_valid.columns:
                df_valid = df_valid[df_valid[col] > 0]

        if len(df_valid) < 50:
            print(f" ! Skipping: Not enough data points ({len(df_valid)} rows).")
            continue

        features = elements + prop_cols
        X = df_valid[features]
        # TARGET IS NOW THE MATHEMATICAL CLUSTER
        y_raw = df_valid['Cluster_ID'].astype(str)

        local_le = LabelEncoder()
        y = local_le.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Tune Models
        rf = RandomForestClassifier(random_state=42)
        rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        rf_search.fit(X_train, y_train)
        acc_rf = accuracy_score(y_test, rf_search.best_estimator_.predict(X_test))

        gb = GradientBoostingClassifier(random_state=42)
        gb_search = RandomizedSearchCV(gb, gb_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        gb_search.fit(X_train, y_train)
        acc_gb = accuracy_score(y_test, gb_search.best_estimator_.predict(X_test))

        xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb_search = RandomizedSearchCV(xgb_clf, xgb_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        xgb_search.fit(X_train, y_train)
        acc_xgb = accuracy_score(y_test, xgb_search.best_estimator_.predict(X_test))

        best_acc = max(acc_rf, acc_gb, acc_xgb)

        if best_acc == acc_rf:
            best_algo, best_params, winning_model = "RandomForest", rf_search.best_params_, rf_search.best_estimator_
        elif best_acc == acc_gb:
            best_algo, best_params, winning_model = "GradientBoosting", gb_search.best_params_, gb_search.best_estimator_
        else:
            best_algo, best_params, winning_model = "XGBoost", xgb_search.best_params_, xgb_search.best_estimator_

        results[set_name] = {'Best Algo': best_algo, 'Accuracy': best_acc, 'Encoder': local_le, 'Model': winning_model, 'Features': prop_cols}
        print(f"   > Winner: {best_algo} | Accuracy: {best_acc:.2%}")

    return results

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    file_name = 'wrought_alloys_final.xlsx'

    base_df, elem_cols, kmeans_model, scaler_model = load_and_preprocess_classification(file_name)

    if base_df is not None:
        print(f"\n\n{'='*30}\n INITIATING RUN (FILTERED: NO 2xxx/7xxx) \n{'='*30}")
        df_filtered = base_df[~base_df['Series'].str.contains('2xxx|7xxx|2000|7000', case=False, na=False)]

        results = train_classification_models_tuned(df_filtered, elem_cols)

        print(f"\n[{'='*20} FINAL SUMMARY {'='*20}]")
        for m, data in results.items():
            print(f"{m}: Best={data['Best Algo']} ({data['Accuracy']:.1%})")

        # =========================================================================
        # LIVE PREDICTION WITH CLUSTER TRANSLATION
        # =========================================================================
        print(f"\n\n{'='*30}\n LIVE PREDICTION DEMONSTRATION (Set 1: YS & EC) \n{'='*30}")

        my_12_elements = {
            'Al': 97.9, 'Si': 0.6, 'Fe': 0.2, 'Cu': 0.1, 'Mn': 0.1, 'Mg': 0.8,
            'Cr': 0.1, 'Ni': 0.0, 'Zn': 0.1, 'Ga': 0.0, 'V': 0.0, 'Ti': 0.1
        }

        # Inject custom YS and EC
        my_custom_input = my_12_elements.copy()
        my_custom_input['YS (MPa)'] = 275.0
        my_custom_input['EC Volume (% IACS)'] = 45.0

        if 'Set 1 (YS & EC)' in results:
            my_input_df = pd.DataFrame([my_custom_input])
            ordered_features = elem_cols + ['YS (MPa)', 'EC Volume (% IACS)']
            my_input_df = my_input_df[ordered_features]

            print("Our Custom Input (12 Metals + YS & EC):")
            print(my_input_df.to_string(index=False))

            winning_model = results['Set 1 (YS & EC)']['Model']
            specific_encoder = results['Set 1 (YS & EC)']['Encoder']
            winning_algo_name = results['Set 1 (YS & EC)']['Best Algo']

            # Predict Cluster
            predicted_encoded = winning_model.predict(my_input_df)
            predicted_cluster = specific_encoder.inverse_transform(predicted_encoded)[0]

            print(f"\n -> Predicted Mathematical Family: {predicted_cluster} (Using {winning_algo_name})")

            # Translate Cluster to Temper
            cluster_tempers = base_df[base_df['Cluster_ID'] == predicted_cluster]['Base_Temper']
            most_common_temper = cluster_tempers.mode()[0] if not cluster_tempers.empty else "Unknown"

            print(f" -> Recommended Manufacturing Temper: {most_common_temper}")