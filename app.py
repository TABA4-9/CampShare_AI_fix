from flask import Flask
from controllers.reco_controller import recommendation_blueprint

app = Flask(__name__)

# Recommendation Controller 등록
app.register_blueprint(recommendation_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
