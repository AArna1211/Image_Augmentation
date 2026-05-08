# 🖼️ Smart Image Augmentation Recommender

A progressively intelligent image augmentation system that moves from fixed pipelines to AI-driven augmentation recommendations — using heuristics, CLIP-based classification, and optionally an LLM for augmentation selection.

---
note: only the scaffolding is up right now, brush ups and deployment on huggingface coming soon.
## 📌 Overview

Rather than blindly applying the same augmentations to every image, this system **analyzes each image first**, then selects the most suitable augmentation strategy. It is designed for ML/CV practitioners who want to improve dataset quality intelligently before training.

The notebook is structured in **four progressive stages**, from simple to AI-powered.

---

## 🧱 Stages

### Stage 1 — Fixed Augmentation Pipeline
A baseline batch augmentation script. Applies a single fixed pipeline to all images in a folder.

**Transforms used:** HorizontalFlip, RandomRotate90, RandomBrightnessContrast, GaussianBlur, RandomCrop, ColorJitter

**Use case:** Quick dataset expansion when you don't need category-awareness.

---

### Stage 2 — Heuristic-Based Smart Augmentation
Adds image quality analysis before augmenting. Uses OpenCV to detect image characteristics and routes to a matching pipeline.

| Condition | Detection Method | Pipeline Applied |
|---|---|---|
| Too dark | Mean pixel brightness < 100 | Brighten + ColorJitter |
| Too blurry | Laplacian variance < 100 | Sharpen + GaussianBlur |
| Low color variance | Pixel variance < 500 | ColorJitter + HueSaturation |
| Normal | None of the above | HorizontalFlip + Rotate90 |

Also includes a **Gradio UI** where you can upload an image and see suggestions + 3 augmented previews interactively.

---

### Stage 3 — CLIP-Based Category Classification + Policy Matching *(Main Version)*
Uses OpenAI's **CLIP** (`clip-vit-base-patch32`) to semantically classify the image into one of four categories, then applies a tailored augmentation policy.

**Categories and their augmentation focus:**

| Category | Key Augmentations | Rationale |
|---|---|---|
| `object` | Rotate90, Flip, ShiftScaleRotate, CLAHE | Objects can appear at any orientation and lighting |
| `person` | Mild rotation, Shadow, Fog, Blur | Preserve body proportions; simulate real-world conditions |
| `animal` | ColorJitter, ElasticTransform, GaussNoise | Fur/texture variation; natural motion effects |
| `text` | Perspective, GridDistortion, Downscale, CLAHE | Simulate scans, camera angles, and print quality |

Includes a **Gradio gallery UI** showing the detected class, confidence score, and 3 augmented previews side-by-side.

---

### Stage 4 — LLM-Guided Augmentation Selection *(Advanced)*
Extends Stage 3 by introducing a **language model** (e.g., Llama-2-7B) as an augmentation policy selector.

**Flow:**
1. CLIP classifies the image → gets category
2. Available augmentations for that category are listed in a prompt
3. LLM reasons and responds with the top-3 most suitable augmentations
4. Those augmentations are applied and previewed

This enables dynamic, reasoning-based selection rather than a fixed per-category policy.

---

## 🚀 Installation

```bash
pip install torch torchvision pillow ftfy regex tqdm transformers albumentations gradio opencv-python
```

For Stage 4 (LLM), additional memory is required for the language model:
```bash
# Llama-2-7B requires ~14GB VRAM or CPU RAM
# Consider using a smaller model for local experimentation
```

---

## 📂 Usage

### Batch mode (Stages 1 & 2)
Place your images in `/content/images/`, then run the relevant stage cell. Augmented images are saved to `/content/augmented_images/`.

```python
input_dir = "/content/images"
output_dir = "/content/augmented_images"
augmentations_per_image = 5  # adjust as needed
```

### Interactive UI (Stages 2, 3, 4)
Run the Gradio cell and open the local URL. Upload any image to get:
- Detected category / quality issue
- Confidence score
- 3 augmented preview images

```python
demo.launch(debug=True)
```

---

## 🔧 Configuration

**Changing categories (Stage 3):**
```python
categories = ["object", "person", "animal", "text"]
```

**Adjusting augmentation probabilities:**
Each transform has a `p=` parameter. Increase toward `1.0` for more aggressive augmentation.

**Swapping the LLM (Stage 4):**
```python
llm_model_name = "your-preferred-model"  # e.g., mistralai/Mistral-7B-v0.1
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch` | Model inference backend |
| `transformers` | CLIP model + LLM loading |
| `albumentations` | Image augmentation pipelines |
| `opencv-python` | Heuristic image analysis |
| `gradio` | Interactive web UI |
| `Pillow` | Image loading and conversion |
| `numpy` | Array operations |

---

## 🗺️ Architecture

```
Input Image
     │
     ▼
[Stage 2] Heuristic Check ──────────────────► Targeted Pipeline
     │         OR
     ▼
[Stage 3] CLIP Classification
     │   object / person / animal / text
     ▼
Category-Specific Augmentation Policy
     │         OR (Stage 4)
     ▼
[Stage 4] LLM Prompt → Top-3 Augmentation Names
     │
     ▼
Apply Augmentations → 3 Preview Images → Gradio Gallery
```

---

## 📝 Notes

- CLIP zero-shot classification works well for broad categories but may misclassify edge cases (e.g., a person holding a large object).
- For production use, consider fine-tuning CLIP on your domain-specific categories.
- The LLM stage (Stage 4) requires significant compute; a GPU with ≥16GB VRAM is recommended.
- All augmentation probabilities are tunable — start conservative (`p=0.3`) and increase based on downstream model performance.

---

## 📄 License

MIT License — free to use, modify, and distribute.
