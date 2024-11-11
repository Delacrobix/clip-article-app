import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, exceptions, helpers

load_dotenv()

ELASTIC_CLUSTER_ID = os.getenv("ES_CLUSTER_ID")
ELASTIC_API_KEY = os.getenv("ES_API_KEY")


es_client = Elasticsearch(
    cloud_id=ELASTIC_CLUSTER_ID,
    api_key=ELASTIC_API_KEY,
)


def index_images(index_name: str, images_obj_arr: list):
    actions = [
        {
            "_index": index_name,
            "_source": obj,
        }
        for obj in images_obj_arr
    ]

    try:
        response = helpers.bulk(es_client, actions)
        return response
    except exceptions.ConnectionError as e:
        return e


def knn_search(index_name: str, query_vector: list, k: int):
    query = {
        "_source": ["image_embedding", "image_name"],
        "query": {
            "knn": {
                "image_embeddings": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    try:
        response = es_client.search(index=index_name, body=query)
        return response
    except exceptions.ConnectionError as e:
        return e
