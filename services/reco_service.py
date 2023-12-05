from .db_service import fetch_user_logs
from models.reco_model import CollaborativeFilteringModel, ContentBasedFilteringModel, hybrid_recommend_products

def get_recommendations(user_id):
    user_logs = fetch_user_logs(user_id)

    # 여기에 상품 데이터 경로를 설정하세요.
    product_data_path = 'path_to_your_product_data.csv'
    cf_model = CollaborativeFilteringModel()
    cb_model = ContentBasedFilteringModel(product_data_path)

    # 마지막으로 조회한 상품 ID (가장 최근의 로그)
    last_viewed_product_id = user_logs[-1]

    # 하이브리드 추천 시스템 실행
    recommendations = hybrid_recommend_products(last_viewed_product_id, last_viewed_product_id, cf_model, cb_model)
    return recommendations
