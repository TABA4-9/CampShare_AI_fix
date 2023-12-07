from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

app = Flask(__name__)

# 전역 변수로 유저 로그와 상품 정보를 저장
user_logs = []
products_info = []

# 협업 필터링 로직
def collaborative_filtering(current_product_id):
    user_logs_df = pd.DataFrame(user_logs)
    user_logs_df['TIMESTAMP'] = pd.to_datetime(user_logs_df['TIMESTAMP'])
    user_logs_df.sort_values(by=['USER_ID', 'TIMESTAMP'], inplace=True)
    user_logs_df['NEXT_PRODUCT_ID'] = user_logs_df.groupby('USER_ID')['PRODUCT_ID'].shift(-1)

    next_products = user_logs_df[user_logs_df['PRODUCT_ID'] == current_product_id]['NEXT_PRODUCT_ID']
    next_product_counts = next_products.value_counts()

    return next_product_counts.to_dict()

# 콘텐츠 기반 필터링 로직
def content_based_filtering(current_product_name):
    products_df = pd.DataFrame(products_info)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(products_df['Product_Name'])

    cosine_sim = cosine_similarity(tfidf_matrix)
    current_product_idx = products_df.index[products_df['Product_Name'] == current_product_name][0]
    similarity_scores = list(enumerate(cosine_sim[current_product_idx]))

    return {products_df.iloc[i[0]]['Product_ID']: i[1] for i in similarity_scores}

# 하이브리드 추천 로직
def hybrid_recommendations(cf_scores, cb_scores):
    combined_scores = defaultdict(float)
    for prod_id, score in cf_scores.items():
        combined_scores[prod_id] += score
    for prod_id, score in cb_scores.items():
        combined_scores[prod_id] += score

    top_recommendations = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
    return top_recommendations

@app.route('/user_logs', methods=['POST'])
def handle_user_logs():
    global user_logs
    user_logs = request.json
    return jsonify({"message": "User logs received"}), 200

@app.route('/product_info', methods=['POST'])
def handle_product_info():
    global products_info
    products_info = request.json
    return jsonify({"message": "Product info received"}), 200

@app.route('/current_product', methods=['POST'])
def handle_current_product():
    current_product = request.json
    current_product_id = current_product['Product_ID']
    current_product_name = current_product['Product_Name']

    cf_scores = collaborative_filtering(current_product_id)
    cb_scores = content_based_filtering(current_product_name)
    combined_recommendations = hybrid_recommendations(cf_scores, cb_scores)

    # 전역 변수 초기화
    user_logs = []
    products_info = []

    return jsonify({"recommended_products": combined_recommendations}), 200

if __name__ == '__main__':
    app.run(debug=True)
