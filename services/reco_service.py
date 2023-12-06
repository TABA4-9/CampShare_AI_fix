from models.reco_model import get_cooccurrence_matrix, get_product_info, get_similarity_df
from utils.reco_utils import cf_recommend_products, recommend_products, hybrid_recommend_products as hybrid_rec

def hybrid_recommend_products(product_id):
    # cooccurrence_matrix 및 기타 필요한 데이터를 가져오는 로직
    cooccurrence_matrix = get_cooccurrence_matrix()
    product_info = get_product_info(product_id)
    similarity_df = get_similarity_df(product_info)

    # 협업 필터링 추천
    cf_recommendations = cf_recommend_products(product_id, cooccurrence_matrix)

    # 콘텐츠 기반 필터링 추천
    cb_recommendations = recommend_products(product_info, similarity_df)

    # 하이브리드 필터링으로 최종 추천 목록 생성
    recommendations = hybrid_rec(cf_recommendations, cb_recommendations)

    return recommendations
