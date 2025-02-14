import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def preprocess_image(image_path):
    """
    Preprocess a CAPTCHA image following these steps:
    1. Convert to float
    2. Convert to grayscale
    3. Normalize to range [-1, 1]
    4. Resize (Downsample)
    """

    # Load the image
    image = Image.open(image_path)

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor (float range 0-1)
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to range [-1, 1]
        transforms.Resize((32, 100))  # Resize image to fixed dimensions
    ])

    # Apply transformations
    processed_image = transform(image)

    return processed_image

def preprocess_dataset(image_folder, output_folder):
    """
    Preprocess all images in a folder and save them as tensors.

    Use .pt if you want preprocessed tensors for fast loading.
    Keep .png if you prefer preprocessing on-the-fly during training.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(image_folder):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_folder, img_name)
            processed_image = preprocess_image(img_path)

            # Save preprocessed tensor
            output_path = os.path.join(output_folder, img_name.replace('.png', '.pt'))  # Here decide file extension
            torch.save(processed_image, output_path)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    dataset_path = r"C:/Users/irene/Documents/UTN/CV/groupAssignment/code/UTN_Captcha_Detector/datasets/utn_dataset_curated/part2/test/images"
    output_path = r"C:/Users/irene/Documents/UTN/CV/groupAssignment/code/UTN_Captcha_Detector/datasets/utn_dataset_curated/part2/test/preprocessed"

    preprocess_dataset(dataset_path, output_path)
