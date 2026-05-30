import albumentations as A

augmentation_policies = {
    "object": A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.MotionBlur(p=0.3),
        A.CLAHE(p=0.3),
    ]),

    "person": A.Compose([
        A.RandomBrightnessContrast(p=0.7),
        A.ColorJitter(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
        A.Resize(256,256,p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
    ]),

    "animal": A.Compose([
        A.RandomRotate90(p=0.7),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomGamma(p=0.5),
        A.MotionBlur(p=0.4),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(p=0.3),
    ]),

    "text": A.Compose([
        A.RandomBrightnessContrast(p=0.7),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.4),
        A.Perspective(scale=(0.05,0.1), p=0.5),
        A.MotionBlur(p=0.3),
        A.CLAHE(p=0.3),
        A.GaussNoise(p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.7, p=0.3),
    ]),
}

# For LLM selector
augmentation_policies_list = {
    "object": [
        "RandomRotate90", "HorizontalFlip", "VerticalFlip", "ShiftScaleRotate",
        "RandomBrightnessContrast", "HueSaturationValue", "MotionBlur", "CLAHE"
    ],
    "person": [
        "RandomBrightnessContrast", "ColorJitter", "Blur", "GaussNoise",
        "Resize", "ShiftScaleRotate", "RandomShadow", "RandomFog"
    ],
    "animal": [
        "RandomRotate90", "HorizontalFlip", "ColorJitter", "RandomBrightnessContrast",
        "RandomGamma", "MotionBlur", "GaussNoise", "ElasticTransform"
    ],
    "text": [
        "RandomBrightnessContrast", "OpticalDistortion", "GridDistortion", "Perspective",
        "MotionBlur", "CLAHE", "GaussNoise", "Downscale"
    ],
}

# Mapping names to albumentations
def map_name_to_augmentation(name: str):
    mapping = {
        "RandomRotate90": A.RandomRotate90(p=1.0),
        "HorizontalFlip": A.HorizontalFlip(p=1.0),
        "VerticalFlip": A.VerticalFlip(p=1.0),
        "ShiftScaleRotate": A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=1.0),
        "RandomBrightnessContrast": A.RandomBrightnessContrast(p=1.0),
        "HueSaturationValue": A.HueSaturationValue(p=1.0),
        "MotionBlur": A.MotionBlur(p=1.0),
        "CLAHE": A.CLAHE(p=1.0),
        "ColorJitter": A.ColorJitter(p=1.0),
        "Blur": A.Blur(blur_limit=3, p=1.0),
        "GaussNoise": A.GaussNoise(var_limit=(10.0,50.0), p=1.0),
        "Resize": A.Resize(256, 256, p=1.0),
        "RandomShadow": A.RandomShadow(p=1.0),
        "RandomFog": A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
        "RandomGamma": A.RandomGamma(p=1.0),
        "ElasticTransform": A.ElasticTransform(p=1.0),
        "OpticalDistortion": A.OpticalDistortion(p=1.0),
        "GridDistortion": A.GridDistortion(p=1.0),
        "Perspective": A.Perspective(scale=(0.05,0.1), p=1.0),
        "Downscale": A.Downscale(scale_min=0.5, scale_max=0.7, p=1.0),
    }
    return mapping[name]