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
            "_source": {
                "image_data": obj["image_data"],
                "image_name": obj["image_name"],
                "image_embedding": obj["image_embedding"],
            },
        }
        for obj in images_obj_arr
    ]

    try:
        return helpers.bulk(es_client, actions)
    except exceptions.ConnectionError as e:
        return e


def knn_search(index_name: str, query_vector: list, k: int):
    query = {
        "size": 4,
        "_source": ["image_name", "image_data"],
        "query": {
            "knn": {
                "field": "image_embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 100,
                "boost": 10,
            }
        },
    }

    try:
        response = es_client.search(index=index_name, body=query)
        return response
    except exceptions.ConnectionError as e:
        return e


def get_all_query(index_name: str):
    query = {
        "size": 400,
        "source": ["image_name", "image_data"],
        "query": {"match_all": {}},
    }

    try:
        return es_client.search(index=index_name, body=query)
    except exceptions.ConnectionError as e:
        return e
