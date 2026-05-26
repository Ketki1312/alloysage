import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/api/load', methods=['POST'])
def load_data():
    # On Vercel serverless, you cannot retrain at runtime.
    # Retrain locally using serialize_models.py and redeploy.
    return jsonify({
        'error': (
            'Live retraining is not supported on Vercel serverless. '
            'Run serialize_models.py locally, commit the new models.pkl, and redeploy.'
        )
    }), 501


handler = app

if __name__ == "__main__":
    app.run(debug=True)