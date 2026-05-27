from flask import Flask
from flask_cors import CORS

from api.health import app as health_app
from api.model_info import app as model_info_app
from api.predict import app as predict_app
from api.predict_all_sets import app as predict_all_sets_app
from api.load_data import app as load_data_app

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "AlloySage Backend Running"}

app.register_blueprint(health_app.blueprints[0] if health_app.blueprints else health_app)
app.register_blueprint(model_info_app.blueprints[0] if model_info_app.blueprints else model_info_app)
app.register_blueprint(predict_app.blueprints[0] if predict_app.blueprints else predict_app)
app.register_blueprint(predict_all_sets_app.blueprints[0] if predict_all_sets_app.blueprints else predict_all_sets_app)
app.register_blueprint(load_data_app.blueprints[0] if load_data_app.blueprints else load_data_app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)