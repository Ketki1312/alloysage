import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from _helpers import models, predict_for_set

app = Flask(__name__)
CORS(app)


@app.route('/api/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Models not loaded. Check models.pkl exists in repo root.'}), 503

    body        = request.get_json(force=True) or {}
    composition = body.get('composition', {})
    set_name    = body.get('set', '')
    properties  = body.get('properties', {})

    if not set_name:
        return jsonify({'error': 'Missing "set" field in request body.'}), 400

    try:
        result = predict_for_set(set_name, composition, properties)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


handler = app

if __name__ == "__main__":
    app.run(debug=True)