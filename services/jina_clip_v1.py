from transformers import AutoModel

model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)


async def generate_image_embedding(image_url: str):
    image_embeddings = model.encode_image([image_url])

    return image_embeddings
