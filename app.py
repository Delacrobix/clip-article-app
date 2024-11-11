import asyncio
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from services.cohere_embed import (
    generate_image_embeddings as cohere_generate_embeddings,
)
from services.jina_clip_v1 import generate_image_embeddings as jina_generate_embeddings
from services.openai_clip import generate_image_embeddings as openai_generate_embeddings

st.title("Image Embedding Generator")


async def fetch_embeddings(image):
    openai_result, cohere_result, jina_result = await asyncio.gather(
        openai_generate_embeddings(image),
        cohere_generate_embeddings(image),
        jina_generate_embeddings(image),
    )

    return openai_result, cohere_result, jina_result


async def main():
    image_url = st.text_input("Enter image URL:")

    if image_url:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))

            st.image(img, caption="Uploaded Image", use_container_width=True)

            openai_result, cohere_result, jina_result = await fetch_embeddings(
                image_url
            )

            st.subheader("Results:")
            st.write("OpenAI Embeddings: ", openai_result)
            st.write("Cohere Embeddings: ", cohere_result)
            st.write("Jina Embeddings: ", jina_result)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
