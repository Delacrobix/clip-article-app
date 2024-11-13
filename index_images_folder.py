import asyncio
import base64
import os

from PIL import Image

from services.cohere_embed import (
    generate_image_embeddings as cohere_generate_embeddings,
)
from services.elasticsearch import index_images
from services.jina_clip_v1 import generate_image_embeddings as jina_generate_embeddings
from services.openai_clip import generate_image_embeddings as openai_generate_embeddings


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


async def main():
    folder_path = "./data"

    jina_index = "jina-images"
    embed_index = "embed-images"
    clip_index = "clip-images"

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

            image_base64 = encode_image_to_base64(img_path)

            jina_obj_arr.append(
                {
                    "image_name": filename,
                    "image_embedding": jina_result,
                    "image_data": image_base64,
                }
            )
            cohere_obj_arr.append(
                {
                    "image_name": filename,
                    "image_embedding": cohere_result,
                    "image_data": image_base64,
                }
            )
            openai_obj_arr.append(
                {
                    "image_name": filename,
                    "image_embedding": openai_result,
                    "image_data": image_base64,
                }
            )

        except Exception as e:
            print(f"Error with {filename}: {e}")

    print("Indexing images in Elasticsearch...")

    jina_response = index_images(jina_index, jina_obj_arr)
    print("Jina response: ", jina_response)

    cohere_response = index_images(embed_index, cohere_obj_arr)
    print("Cohere response: ", cohere_response)

    openai_response = index_images(clip_index, openai_obj_arr)
    print("OpenAI response: ", openai_response)


asyncio.run(main())
