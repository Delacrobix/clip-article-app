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


async def generate_text_embeddings(description: str):
    try:
        text_embeddings = model.encode_text(description)

        return text_embeddings.tolist()

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
