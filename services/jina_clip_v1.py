import torch
from PIL import Image
from transformers import AutoModel

model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)


async def generate_image_embeddings(image: Image.Image):
    try:
        image_embeddings = model.encode_image([image])

        return image_embeddings[0].tolist()

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
