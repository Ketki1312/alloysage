import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify
from flask_cors import CORS
from _helpers import models

app = Flask(__name__)
CORS(app)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    info = {}
    for name, m in models.items():
        info[name] = {
            'algorithm': m['algorithm'],
            'accuracy':  m['accuracy'],
            'features':  m['features'],
        }
    return jsonify(info)


handler = app

if __name__ == "__main__":
    app.run(debug=True)