import pyodbc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def db_connection():
    return pyodbc.connect(
        'MYDB;UID=sys;PWD=tibero')


def get_cooccurrence_matrix():
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_logs")
    logs = cursor.fetchall()
    conn.close()

    # 로그 데이터를 기반으로 cooccurrence_matrix 생성
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
    for row in logs:
        viewed_products = row[1].split(', ')
        for i in range(len(viewed_products)):
            for j in range(i + 1, len(viewed_products)):
                product1, product2 = viewed_products[i], viewed_products[j]
                cooccurrence_matrix[product1][product2] += 1
                cooccurrence_matrix[product2][product1] += 1
    return cooccurrence_matrix


def get_product_info(product_id):
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT NAME FROM products WHERE id=?", (product_id,))
    product_info = cursor.fetchone()
    conn.close()
    return product_info


def get_similarity_df(product_info):
    # DB에서 전체 상품 데이터를 가져와서 TF-IDF와 코사인 유사도 계산
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT NAME FROM products")
    products = cursor.fetchall()
    conn.close()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([product_info[0]] + [p[0] for p in products])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return pd.DataFrame(cosine_sim, index=[product_info[0]] + [p[0] for p in products],
                        columns=[product_info[0]] + [p[0] for p in products])
