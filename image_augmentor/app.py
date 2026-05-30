import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import albumentations as A
from core.clip_classifier import classify_image
from core.llm_selector import llm_suggest_augmentations
from pipelines.augmentation_policies import augmentation_policies, augmentation_policies_list, map_name_to_augmentation

# Padding helper
def pad_image(img: Image.Image, target_size=(512, 512), color=(0, 0, 0)):
    img = img.convert("RGB")
    w, h = img.size
    new_w, new_h = target_size
    delta_w = max(new_w - w, 0)
    delta_h = max(new_h - h, 0)
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding, fill=color)

# Apply selected augmentations
def apply_llm_augmentations(img, augmentation_names):
    np_img = np.array(img.convert("RGB"))
    applied_imgs = []
    for aug_name in augmentation_names:
        aug = map_name_to_augmentation(aug_name)
        composed = A.Compose([aug])
        applied_imgs.append(Image.fromarray(composed(image=np_img)["image"]))
    return applied_imgs

# Gradio function
def process_image(img):
    if img is None:
        return "No image uploaded.", None

    # Step 0: pad
    padded_img = pad_image(img, target_size=(512,512), color=(0,0,0))

    # Step 1: classify
    category, probs = classify_image(padded_img)

    # Step 2: LLM suggests top 3 augmentations
    available_ops = augmentation_policies_list[category]
    top_augs = llm_suggest_augmentations(category, available_ops, top_k=3)

    # Step 3: apply augmentations
    previews = apply_llm_augmentations(padded_img, top_augs)

    # Step 4: output
    suggestion_text = f"Detected category: **{category}** (confidence {probs[category]:.2f})\n"
    suggestion_text += f"LLM suggested augmentations: {', '.join(top_augs)}"
    return suggestion_text, previews

# Gradio UI
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Markdown(label="Detected Class & Suggested Augmentations"),
        gr.Gallery(label="Augmented Previews", columns=3, height="auto")
    ],
    title="Class-Based LLM Augmentation Recommender",
    description="Upload an image. CLIP detects its class, then LLM recommends top 3 augmentations from that class. Previews are shown."
)

if __name__ == "__main__":
    demo.launch(debug=True)