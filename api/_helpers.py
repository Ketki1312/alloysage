"""
Shared helpers loaded by every API serverless function.
Models are loaded from models.pkl (pre-trained locally via serialize_models.py).
"""

import os
import re
import pickle
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

ELEMENTS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']

# ── Resolve paths relative to repo root ───────────────────────────────────────
# api/_helpers.py lives in  /repo/api/
# models.pkl lives in       /repo/models.pkl
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
PKL_PATH  = os.path.join(_ROOT, 'models.pkl')
DATA_PATH = os.path.join(_ROOT, 'wrought_alloys_final.xlsx')

# ── Global state ──────────────────────────────────────────────────────────────
models        = {}
base_df       = None
kmeans_model  = None
scaler_model  = None
load_error    = ""

# ── Load from pickle ──────────────────────────────────────────────────────────
def _load():
    global models, base_df, kmeans_model, scaler_model, load_error
    try:
        with open(PKL_PATH, 'rb') as f:
            saved = pickle.load(f)
        models       = saved['models']
        base_df      = saved['base_df']
        kmeans_model = saved.get('kmeans_model')
        scaler_model = saved.get('scaler_model')
    except Exception as e:
        load_error = str(e)

_load()

# ── Utility ───────────────────────────────────────────────────────────────────
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


def predict_for_set(set_name, composition, properties):
    """
    Returns a dict with cluster, recommended_temper, temper_distribution,
    algorithm, accuracy, and probabilities.
    Raises ValueError on bad input.
    """
    if set_name not in models:
        raise ValueError(f"Unknown set '{set_name}'. Valid: {list(models.keys())}")

    m = models[set_name]
    prop_cols = m['features']

    row = {el: float(composition.get(el, 0.0)) for el in ELEMENTS}
    for col in prop_cols:
        if col not in properties:
            raise ValueError(f"Missing property '{col}'")
        row[col] = float(properties[col])

    input_df = pd.DataFrame([row])[ELEMENTS + prop_cols]

    encoded = m['model'].predict(input_df)
    cluster = m['encoder'].inverse_transform(encoded)[0]

    most_common = 'Unknown'
    temper_dist = {}
    if base_df is not None:
        cluster_tempers = base_df[base_df['Cluster_ID'] == cluster]['Base_Temper']
        if not cluster_tempers.empty:
            most_common = cluster_tempers.mode()[0]
        temper_dist = cluster_tempers.value_counts().to_dict()

    proba = None
    if hasattr(m['model'], 'predict_proba'):
        try:
            proba_arr = m['model'].predict_proba(input_df)[0]
            labels = m['encoder'].inverse_transform(range(len(proba_arr)))
            cluster_temper_map = {}
            if base_df is not None:
                for c_id in labels:
                    ct = base_df[base_df['Cluster_ID'] == c_id]['Base_Temper']
                    cluster_temper_map[c_id] = ct.mode()[0] if not ct.empty else 'Unknown'
            proba = [
                {
                    'cluster': labels[i],
                    'temper': cluster_temper_map.get(labels[i], '?'),
                    'probability': round(float(p), 4)
                }
                for i, p in enumerate(proba_arr)
            ]
            proba.sort(key=lambda x: x['probability'], reverse=True)
        except Exception:
            pass

    return {
        'cluster': cluster,
        'recommended_temper': most_common,
        'temper_distribution': temper_dist,
        'algorithm': m['algorithm'],
        'accuracy': m['accuracy'],
        'probabilities': proba,
    }