from flask import Flask, request, jsonify
from flask_cors import CORS

from api._helpers import models, load_error, predict_for_set

app = Flask(__name__)
CORS(app)


# ───────────────── HEALTH ─────────────────

@app.route('/api/health', methods=['GET'])
def health():
    ok = bool(models) and not load_error

    return jsonify({
        'status': 'ok' if ok else 'degraded',
        'models_loaded': list(models.keys()),
        'error': load_error if not ok else None,
    })


# ───────────────── MODEL INFO ─────────────────

@app.route('/api/model-info', methods=['GET'])
def model_info():
    info = {}

    for name, m in models.items():
        info[name] = {
            'algorithm': m['algorithm'],
            'accuracy': m['accuracy'],
            'features': m['features'],
        }

    return jsonify(info)


# ───────────────── PREDICT ─────────────────

@app.route('/api/predict', methods=['POST'])
def predict():

    if not models:
        return jsonify({
            'error': 'Models not loaded'
        }), 503

    body = request.get_json(force=True) or {}

    composition = body.get('composition', {})
    set_name = body.get('set', '')
    properties = body.get('properties', {})

    if not set_name:
        return jsonify({
            'error': 'Missing set field'
        }), 400

    try:
        result = predict_for_set(
            set_name,
            composition,
            properties
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# ───────────────── PREDICT ALL SETS ─────────────────

@app.route('/api/predict-all-sets', methods=['POST'])
def predict_all_sets():

    body = request.get_json(force=True) or {}

    composition = body.get('composition', {})
    properties = body.get('properties', {})

    results = {}

    for set_name in models.keys():

        try:
            results[set_name] = predict_for_set(
                set_name,
                composition,
                properties
            )

        except Exception as e:
            results[set_name] = {
                'error': str(e)
            }

    return jsonify(results)


# ───────────────── LOAD ─────────────────

@app.route('/api/load', methods=['POST'])
def load():

    return jsonify({
        'message': 'Models already loaded from models.pkl'
    })


# ───────────────── HOME ─────────────────

@app.route('/')
def home():
    return {
        "message": "AlloySage API Running"
    }


# ───────────────── MAIN ─────────────────

if __name__ == '__main__':
    app.run(debug=True)