def cf_recommend_products(product_id, cooccurrence_matrix, top_n=5):
    # 협업 필터링 로직
    related_products = cooccurrence_matrix[product_id]
    recommended = sorted(related_products.items(), key=lambda x: x[1], reverse=True)
    return [prod for prod, _ in recommended[:top_n]]

def recommend_products(product_info, similarity_df, top_n=5):
    # 콘텐츠 기반 필터링 로직
    sim_scores = similarity_df[product_info]
    sorted_indices = sim_scores.argsort()[::-1]
    return similarity_df.columns[sorted_indices][1:top_n + 1]

def hybrid_recommend_products(cf_recommendations, cb_recommendations, top_n=5):
    # 하이브리드 필터링 로직
    combined_recommendations = list(set(cf_recommendations + cb_recommendations))
    return combined_recommendations[:top_n]
