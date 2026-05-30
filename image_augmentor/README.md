# Image Augmentor

A smart image augmentation tool that uses heuristics, CLIP classification, and LLM to suggest and apply augmentations.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Structure

- `app.py`: Main Gradio application.
- `core/`: Core modules for heuristics, CLIP, LLM.
- `pipelines/`: Augmentation policies.
- `configs/`: Configuration files.