from db_service import fetch_user_logs, fetch_product_data
from models.reco_model import CollaborativeFilteringModel, ContentBasedFilteringModel, hybrid_recommend_products

def get_recommendations(user_id):
    user_logs = fetch_user_logs(user_id)
    product_data = fetch_product_data()

    cf_model = CollaborativeFilteringModel(user_logs)
    cb_model = ContentBasedFilteringModel(product_data)

    # 마지막으로 조회한 상품 ID (가장 최근의 로그)
    last_viewed_product_id = user_logs[-1] if user_logs else None

    # 하이브리드 추천 시스템 실행
    if last_viewed_product_id:
        recommendations = hybrid_recommend_products(last_viewed_product_id, cf_model, cb_model)
    else:
        recommendations = []

    return recommendations
