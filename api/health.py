import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify
from flask_cors import CORS
from _helpers import models, load_error

app = Flask(__name__)
CORS(app)


@app.route('/api/health', methods=['GET'])
def health():
    ok = bool(models) and not load_error
    return jsonify({
        'status': 'ok' if ok else 'degraded',
        'models_loaded': list(models.keys()),
        'error': load_error if not ok else None,
    })


# Vercel entry point
handler = app

if __name__ == "__main__":
    app.run(debug=True)