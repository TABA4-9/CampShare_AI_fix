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
    similarity_scores = list(enumerate(cosine_sim[current_product_idx]))

    return {products_df.iloc[i[0]]['id']: i[1] for i in similarity_scores}

# 하이브리드 추천 로직
def hybrid_recommendations(cf_scores, cb_scores):
    combined_scores = defaultdict(float)
    for prod_id, score in cf_scores.items():
        combined_scores[prod_id] += score
    for prod_id, score in cb_scores.items():
        combined_scores[prod_id] += score

    top_recommendations = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
    # int64 타입을 기본 int 타입으로 변환
    return [int(prod_id) for prod_id in top_recommendations]

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
        # 에러 처리 로직, 예를 들어 오류 메시지 반환
        return jsonify({'error': 'Missing id in current_product'}), 400
    if 'name' in current_product:
        current_product_name = current_product['name']
    else:
        # 에러 처리 로직, 예를 들어 오류 메시지 반환
        return jsonify({'error': 'Missing id in current_product'}), 400


    cf_scores = collaborative_filtering(current_product_id)
    cb_scores = content_based_filtering(current_product_name)
    combined_recommendations = hybrid_recommendations(cf_scores, cb_scores)

    # 전역 변수 초기화
    user_logs = []
    products_info = []

    # 반환할 JSON 객체 생성
    recommendations_json = {}
    for rank, product_id in enumerate(combined_recommendations, start=1):
        recommendations_json[f'recommendItemId{rank}'] = product_id

    return jsonify(recommendations_json), 200

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
