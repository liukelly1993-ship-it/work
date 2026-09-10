# 将业务库中的数据同步到Milvus中
from pymilvus import MilvusClient, DataType
from pathlib import Path
import pymysql
from pymysql.cursors import DictCursor
import sys
ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH)) #将项目根目录添加到系统中

from tools.milvus_utils import get_client, get_embedding
from dotenv import load_dotenv
import os

load_dotenv()
# 添加向量数据


def insert_data():
    client = get_client()
    collection_name = os.getenv("COLLECTION_NAME")
    
    # 判断集合是否存在，如果存在则删除
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 定义集合结构
    schema = (MilvusClient.create_schema(auto_id=True)
            .add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            .add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
            .add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    )
    
    # 定义索引类型
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="IP" #如果embedding已经进行了归一化，IP效率高 
    )

    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )    
    
    # 要从业务库中检索了
    with pymysql.connect(
            host=os.getenv("HOST"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            database=os.getenv("DATABASE"),
            port=int(os.getenv("PORT")),
            charset=os.getenv("CHARSET")
        ) as conn: # type:ignore
            with conn.cursor(DictCursor) as cursor:
                cursor.execute("""  
                    select 
                        concat(
                            '菜品名称:', ifnull(dish_name, ''), ';' , 
                            '价格:', ifnull(price, ''), ';',
                            '描述:', ifnull(description, ''), ';',
                            '菜品类别:', ifnull(category, ''), ';',
                            '麻辣程度:', ifnull(spice_level, ''), ';',
                            '口味:', ifnull(flavor, ''), ';',
                            '主料:', ifnull(main_ingredients, ''), ';',
                            '烹饪方法:', ifnull(cooking_method, ''), ';',
                            '是否素食:', ifnull(is_vegetarian, ''), ';',
                            '过敏源:', ifnull(allergens, '')
                        ) as text
                    from menu_items 
                """)
                rows = cursor.fetchall() #[{菜品名称:'', 价格:''}, {菜品名称:''}]
                main_dishes = [row['text'] for row in rows]
    
    # 需要将每个条记录（{菜品名称:'', 价格:''}）转换为字符串（"{菜品名称:'', 价格:''}"）
    #print(main_dishes)
    
    # id\vector\text
    # 生成向量值
    embeddings = get_embedding()
    vector_list = embeddings.embed_documents(main_dishes)
   
    # 构建数据结构
    data_to_insert=[
        {"vector":vec, "text": txt} for vec, txt in zip(vector_list, main_dishes)
    ]
    res = client.insert(collection_name=collection_name, data=data_to_insert)
    print(res)

    
if __name__ == '__main__':
    insert_data()