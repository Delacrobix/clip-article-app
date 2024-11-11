import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


async def generate_image_embedding(image_url: str):
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt", padding=True)

    outputs = model.get_image_features(**inputs)
    return outputs
