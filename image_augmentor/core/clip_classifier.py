import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

categories = ["object", "person", "animal", "text"]

def classify_image(img: Image.Image):
    inputs = processor(text=categories, images=img, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
    return categories[np.argmax(probs)], dict(zip(categories, probs))