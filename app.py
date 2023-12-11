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
    if not user_logs or current_product_id not in [log['itemId'] for log in user_logs]:
        return {}
    user_logs_df = pd.DataFrame(user_logs)
    user_logs_df['TIMESTAMP'] = pd.to_datetime(user_logs_df['timeStamp'])
    user_logs_df.sort_values(by=['userId', 'timeStamp'], inplace=True)
    user_logs_df['NEXT_PRODUCT_ID'] = user_logs_df.groupby('userId')['itemId'].shift(-1)

    next_products = user_logs_df[user_logs_df['itemId'] == current_product_id]['NEXT_PRODUCT_ID']
    next_product_counts = next_products.value_counts()

    return next_product_counts.astype(int).to_dict()


# 콘텐츠 기반 필터링 로직
def content_based_filtering(current_product_name):
    products_df = pd.DataFrame(products_info)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(products_df['name'])

    cosine_sim = cosine_similarity(tfidf_matrix)
    current_product_idx = products_df.index[products_df['name'] == current_product_name][0]

    # 현재 상품에 대한 유사도 점수를 0으로 설정
    cosine_sim[current_product_idx] = 0

    similarity_scores = list(enumerate(cosine_sim[current_product_idx]))

    # 현재 상품을 제외한 다른 상품들에 대한 유사도 점수 반환
    filtered_scores = {products_df.iloc[i[0]]['id']: i[1] for i in similarity_scores}

    return filtered_scores




# 하이브리드 추천 로직
def hybrid_recommendations(current_product_id, current_product_name, top_n=3):
    # 협업 필터링 기반 점수
    cf_scores = collaborative_filtering(current_product_id)
    cf_ranked_scores = {prod_id: top_n - rank for rank, prod_id in enumerate(sorted(cf_scores, key=cf_scores.get, reverse=True)[:top_n])}

    # 콘텐츠 기반 필터링 기반 점수
    cb_scores = content_based_filtering(current_product_name)
    cb_ranked_scores = {prod_id: top_n - rank for rank, prod_id in enumerate(sorted(cb_scores, key=cb_scores.get, reverse=True)[:top_n])}

    # 협업 필터링 결과만 사용하는 경우
    if not cb_ranked_scores:
        return sorted(cf_scores, key=cf_scores.get, reverse=True)[:top_n]

    # 콘텐츠 기반 필터링 결과만 사용하는 경우
    if not cf_ranked_scores:
        return sorted(cb_scores, key=cb_scores.get, reverse=True)[:top_n]

    # 점수 기반 결과 결합
    combined_scores = defaultdict(int)
    for prod_id in set(cf_ranked_scores.keys()).union(cb_ranked_scores.keys()):
        combined_scores[prod_id] += cf_ranked_scores.get(prod_id, 0) + cb_ranked_scores.get(prod_id, 0)

    # 상위 N개 추천 반환
    return [int(prod_id) for prod_id, score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]


@app.route('/test/product', methods=['POST'])
def handle_product_info():
    global products_info
    products_info = request.json
    return ('', 204)

@app.route('/test/log', methods=['POST'])
def handle_user_logs():
    global user_logs
    user_logs = request.json
    return ('', 204)

@app.route('/test/search', methods=['POST'])
def handle_current_product():
    current_product = request.json
    if 'id' in current_product:
        current_product_id = current_product['id']
    else:
        return jsonify({'error': 'Missing id in current_product'}), 400
    if 'name' in current_product:
        current_product_name = current_product['name']
    else:
        return jsonify({'error': 'Missing name in current_product'}), 400

    combined_recommendations = hybrid_recommendations(current_product_id, current_product_name)

    # 반환할 JSON 객체 생성
    recommendations_json = {}
    for rank, product_id in enumerate(combined_recommendations, start=1):
        recommendations_json[f'recommendItemId{rank}'] = product_id

    return jsonify(recommendations_json), 200

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
