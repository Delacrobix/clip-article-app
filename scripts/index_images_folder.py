import asyncio
import os

from PIL import Image

from services.cohere_embed import (
    generate_image_embeddings as cohere_generate_embeddings,
)
from services.jina_clip_v1 import generate_image_embeddings as jina_generate_embeddings
from services.openai_clip import generate_image_embeddings as openai_generate_embeddings


async def main():
    folder_path = "./data"
    jina_obj_arr = []
    cohere_obj_arr = []
    openai_obj_arr = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        try:
            image_data = Image.open(img_path)

            print(f"Loading {filename}...")

            openai_result, cohere_result, jina_result = await asyncio.gather(
                openai_generate_embeddings(image_data),
                cohere_generate_embeddings(image_data),
                jina_generate_embeddings(image_data),
            )

            jina_obj_arr.append(
                {"image_name": filename, "image_embeddings": jina_result}
            )
            cohere_obj_arr.append(
                {"image_name": filename, "image_embeddings": cohere_result}
            )
            openai_obj_arr.append(
                {"image_name": filename, "image_embeddings": openai_result}
            )

        except Exception as e:
            print(f"Error al abrir {filename}: {e}")


asyncio.run(main())
