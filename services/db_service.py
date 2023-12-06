import pyodbc


def get_db_connection():
    connection_string = 'DSN=MYDB;UID=sys;PWD=tibero'  # 데이터베이스 연결 설정
    return pyodbc.connect(connection_string)


def fetch_user_logs(user_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT ITEM_ID FROM VIEW_LOG WHERE USER_ID = ? ORDER BY TIME_STAMP"  # 적절한 쿼리로 변경
    cursor.execute(query, user_id)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return [row[0] for row in rows]


def fetch_product_data():
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT Name FROM product"  # 'products'는 상품 데이터가 저장된 테이블 이름
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    # 데이터프레임 형식으로 변환
    columns = ['Name']  # 실제 데이터베이스 컬럼에 맞게 수정
    return pd.DataFrame(rows, columns=columns)
