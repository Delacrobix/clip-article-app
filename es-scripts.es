PUT clip-images
{
  "mappings": {
    "properties": {
      "image_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "image_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": "true",
        "similarity": "cosine"
      }
    }
  }
}

PUT embed-images
{
  "mappings": {
    "properties": {
      "image_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "image_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": "true",
        "similarity": "cosine"
      }
    }
  }
}

PUT jina-images
{
  "mappings": {
    "properties": {
      "image_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "image_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": "true",
        "similarity": "cosine"
      }
    }
  }
}

GET jina-images/_search
{
  "_source": ["image_embedding", "image_name"],
  "knn": {
    "field": "image_embedding",
    "k": 10,
    "num_candidates": 100,
    "query_vector": [],
    "boost": 10
  }
}