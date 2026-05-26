import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from _helpers import models, predict_for_set

app = Flask(__name__)
CORS(app)


@app.route('/api/predict-all-sets', methods=['POST'])
def predict_all_sets():
    if not models:
        return jsonify({'error': 'Models not loaded.'}), 503

    body        = request.get_json(force=True) or {}
    composition = body.get('composition', {})
    properties  = body.get('properties', {})

    results = {}
    for set_name, m in models.items():
        missing = [c for c in m['features'] if c not in properties]
        if missing:
            results[set_name] = {'error': f'Missing properties: {missing}'}
            continue
        try:
            results[set_name] = predict_for_set(set_name, composition, properties)
        except Exception as e:
            results[set_name] = {'error': str(e)}

    return jsonify(results)


handler = app

if __name__ == "__main__":
    app.run(debug=True)