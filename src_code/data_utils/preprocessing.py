import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import yaml
import sys


def load_config(config_path):
    """Loads the YAML configuration file."""
    config_path = str(config_path)  # Ensure config path is a string
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def preprocess_image(image_path, downscale_factor=None, mean=0.5, std=0.5):
    """
    Preprocess a CAPTCHA image following these steps:
    1. Convert to tensor (float values)
    2. Convert to grayscale (single channel)
    3. Normalize using provided mean and std deviation (range [-1, 1])
    4. Resize using downscale factor or fixed dimensions (32x100)
    """

    # Load the image
    image = Image.open(image_path).convert("RGB")  # Convert RGBA to RGB

    # Create resize transform based on downscale factor
    if downscale_factor:
        resize_transform = transforms.Resize(
            (int(image.height / downscale_factor), int(image.width / downscale_factor))
        )
    else:
        resize_transform = transforms.Resize((32, 100))

    # Define transformation pipeline
    transform = transforms.Compose([
        resize_transform,  # Apply resizing
        transforms.ToTensor(),  # Convert image to tensor (float range 0-1)
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Normalize(mean=[mean], std=[std])  # Normalize to range [-1, 1]
    ])
    return transform(image)


def get_img_transform(configs):
     transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor (float range 0-1)
        transforms.Normalize(mean = [0.4429, 0.5330, 0.4279], std = [0.0841, 0.0718, 0.0796]),
        transforms.Resize((configs.img_height, configs.img_width)) # Normalize to range [-1, 1]
    ])
     return transform


def get_rectangle_img_transform(configs):
     transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor (float range 0-1)
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        transforms.Resize((configs.img_height//configs.downscale_factor, configs.img_width//configs.downscale_factor)) # Normalize to range [-1, 1]
    ])
     return transform


def preprocess_bounding_boxes(bbox, downscale_factor=None):
    """
    Adjust bounding box coordinates if a downscale factor is provided.
    Bounding box format: [x_center, y_center, width, height]
    """
    if downscale_factor:
        x_center, y_center, width, height = bbox
        return [
            x_center / downscale_factor,
            y_center / downscale_factor,
            width / downscale_factor,
            height / downscale_factor
        ]
    return bbox


def preprocess_dataset(image_folder, output_folder, downscale_factor=None, mean=0.5, std=0.5):
    """
    Preprocess all images in a folder and save them as tensors.

    Use .pt if you want preprocessed tensors for fast loading.
    Keep .png if you prefer preprocessing on-the-fly during training.
    """
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"Preprocessed data already exists at {output_folder}. Skipping preprocessing.")
        return  # Skip processing if preprocessed data is found

    print(f"Preprocessing dataset and saving to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    for img_name in os.listdir(image_folder):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_folder, img_name)
            processed_image = preprocess_image(img_path, downscale_factor, mean, std)
            output_path = os.path.join(output_folder, img_name.replace('.png', '.pt'))
            torch.save(processed_image, output_path)
            print(f"Saved preprocessed image to: {output_path}")


def deprocess_image(tensor, mean=0.5, std=0.5):
    """
    Convert a normalized tensor back to an image for visualization or saving.
    """
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-mean / std], std=[1 / std]),  # Reverse normalization
        transforms.ToPILImage()  # Convert tensor to PIL Image
    ])
    return inv_transform(tensor)


def preprocess_all(config_path):
    """Preprocess train, validation, and test datasets only if necessary."""
    config = load_config(config_path)  # Load the YAML config

    script_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = {
        "train": {
            "image_path": os.path.join(script_dir, '..', '..', config['data_configs']['train_path'], 'images'),
            "output_path": os.path.join(script_dir, '..', '..', config['data_configs']['train_path'], 'preprocessed')
        },
        "val": {
            "image_path": os.path.join(script_dir, '..', '..', config['data_configs']['val_path'], 'images'),
            "output_path": os.path.join(script_dir, '..', '..', config['data_configs']['val_path'], 'preprocessed')
        },
        "test": {
            "image_path": os.path.join(script_dir, '..', '..', config['data_configs']['test_path'], 'images'),
            "output_path": os.path.join(script_dir, '..', '..', config['data_configs']['test_path'], 'preprocessed')
        }
    }

    for dataset, paths in datasets.items():
        print(f"\nChecking {dataset} dataset preprocessing...")
        preprocess_dataset(
            paths["image_path"],
            paths["output_path"],
            downscale_factor=config['data_configs']['preprocessing_related'].get('downscale_factor', None),
            mean=config['data_configs']['preprocessing_related']['mean'],
            std=config['data_configs']['preprocessing_related']['std']
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src_code/data_utils/preprocessing.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]  # Get config path from command line
    preprocess_all(config_path)
