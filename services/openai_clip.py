from typing import List

import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


async def compare_image_with_text(image_url: str, texts: List[str]):
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs[0].item()
