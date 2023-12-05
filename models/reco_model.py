from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.db_service import fetch_product_data


# 협업 필터링 모델
class CollaborativeFilteringModel:
    def __init__(self):
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

    def update_cooccurrence_matrix(self, user_log):
        viewed_products = user_log.split(', ')
        for i in range(len(viewed_products)):
            for j in range(i + 1, len(viewed_products)):
                product1, product2 = viewed_products[i], viewed_products[j]
                self.cooccurrence_matrix[product1][product2] += 1
                self.cooccurrence_matrix[product2][product1] += 1

    def recommend_products(self, viewed_product, top_n=5):
        related_products = self.cooccurrence_matrix[viewed_product]
        recommended = sorted(related_products.items(), key=lambda x: x[1], reverse=True)
        return [product for product, _ in recommended[:top_n]]

# 콘텐츠 기반 필터링 모델
class ContentBasedFilteringModel:
    def __init__(self):
        self.df = fetch_product_data()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['Name'])  # 'Name' 컬럼은 상품명을 나타냅니다.
        self.similarity_df = pd.DataFrame(cosine_similarity(self.tfidf_matrix), index=self.df['Name'], columns=self.df['Name'])

    def recommend_products(self, selected_product, top_n=5):
        if selected_product not in self.similarity_df.columns:
            return ["선택한 상품이 데이터에 없습니다."]
        sim_scores = self.similarity_df[selected_product]
        sorted_indices = np.argsort(-sim_scores)
        recommended_products_list = []
        for index in sorted_indices:
            product = self.similarity_df.index[index]
            if product != selected_product and product not in recommended_products_list:
                recommended_products_list.append(product)
                if len(recommended_products_list) >= top_n:
                    break
        return recommended_products_list

# 하이브리드 추천 모델
def hybrid_recommend_products(user_viewed_product, selected_product, cf_model, cb_model, top_n=5):
    cf_recommendations = cf_model.recommend_products(user_viewed_product, top_n)
    cb_recommendations = cb_model.recommend_products(selected_product, top_n)

    combined_scores = defaultdict(int)
    for rank, product in enumerate(cf_recommendations):
        combined_scores[product] += top_n - rank
    for rank, product in enumerate(cb_recommendations):
        combined_scores[product] += top_n - rank

    return [prod for prod, _ in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]

# 사용 예시
# product_data_path는 Tibero DB에서 추출한 상품 데이터 파일의 경로를 나타냅니다.
cf_model = CollaborativeFilteringModel()
cb_model = ContentBasedFilteringModel()

# 사용자 로그 업데이트
user_log = '145, 146, 147'  # 예시 로그 데이터
cf_model.update_cooccurrence_matrix(user_log)

# 하이브리드 추천 실행
user_viewed_product = '145'
selected_product = '네이처하이크 에어텐트 12X'
recommendations = hybrid_recommend_products(user_viewed_product, selected_product, cf_model, cb_model)
