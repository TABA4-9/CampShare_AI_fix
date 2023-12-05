from flask import Flask
from controllers.reco_controller import recommendation_blueprint

app = Flask(__name__)
app.register_blueprint(recommendation_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
