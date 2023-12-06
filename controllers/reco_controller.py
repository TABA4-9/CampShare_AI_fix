from flask import Blueprint, request, jsonify
from services.reco_service import hybrid_recommend_products

recommendation_blueprint = Blueprint('recommendation', __name__)

@recommendation_blueprint.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_id = data.get('product_id')
    recommendations = hybrid_recommend_products(product_id)
    return jsonify({'recommendations': recommendations})
