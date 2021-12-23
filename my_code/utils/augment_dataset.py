import albumentations as A
import numpy as np
import cv2

# Declare an augmentation pipeline

transform = A.Compose([
    A.RandomCrop(width=128, height=128, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    #A.Blur (blur_limit=5, always_apply=False, p=0.5),
    #A.ColorJitter(p=0.2),
    #A.Downscale(p=0.2),
    #A.ISONoise(p=0.2), 
    A.RandomShadow(p=0.2),
    A.ToGray(p=0.2),
    A.ShiftScaleRotate(p=0.3)
])

def augment(image, bgr=True, img_size=(224,224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    if bgr:
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    if img_size:
        transformed_image = cv2.resize(transformed_image, img_size)

    return transformed_image