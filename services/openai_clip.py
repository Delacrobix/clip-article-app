from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


async def generate_image_embeddings(image: Image.Image):
    try:
        inputs = processor(images=image, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)

        return outputs.detach().cpu().numpy().flatten().tolist()

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
