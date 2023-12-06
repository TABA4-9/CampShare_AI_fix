from flask import Blueprint, request, jsonify
from services.reco_service import get_recommendations

recommendation_blueprint = Blueprint('recommendation', __name__)

@recommendation_blueprint.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_id = data.get('product_id')
    recommendations = get_recommendations(product_id)
    return jsonify(recommendations)
