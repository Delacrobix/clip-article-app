import base64
import io
import os

import cohere
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


co = cohere.ClientV2(COHERE_API_KEY)


async def generate_image_embeddings(image: Image.Image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        stringified_buffer = base64.b64encode(img_byte_arr).decode("utf-8")
        content_type = "image/jpeg"
        image_base64 = f"data:{content_type};base64,{stringified_buffer}"

        response = co.embed(
            model="embed-english-v3.0",
            input_type="image",
            embedding_types=["float"],
            images=[image_base64],
        )

        return response.embeddings.float_[0]

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None
