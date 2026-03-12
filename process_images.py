import os
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def process_images(source_dir, dest_dir, num_samples=1000, target_size=(224, 224)):
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Get all jpg files in the source directory
    all_images = [
        f
        for f in os.listdir(source_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ]

    print(f"Found {len(all_images)} images in {source_dir}")

    # Randomly sample 1000 images (or all if less than 1000)
    samples = all_images[: min(num_samples, len(all_images))]

    print(f"Sampling {len(samples)} images...")

    for img_name in tqdm(samples, desc="Processing images"):
        img_path = os.path.join(source_dir, img_name)
        dest_path = os.path.join(dest_dir, img_name)

        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary (e.g. grayscale or RGBA)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize and crop (standard center crop)
                # 1. Resize the shorter side to 256
                width, height = img.size
                if width < height:
                    new_width = 256
                    new_height = int(height * (256 / width))
                else:
                    new_height = 256
                    new_width = int(width * (256 / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 2. Center crop to 224x224
                left = (new_width - target_size[0]) / 2
                top = (new_height - target_size[1]) / 2
                right = (new_width + target_size[0]) / 2
                bottom = (new_height + target_size[1]) / 2

                img = img.crop((left, top, right, bottom))

                # Save the processed image
                img.save(dest_path)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")


if __name__ == "__main__":
    SOURCE_DIR = r"c:\Schoolwork\CSE 493\Project\00"
    DEST_DIR = r"c:\Schoolwork\CSE 493\Final Project\osv5m_sampled_1000"

    process_images(SOURCE_DIR, DEST_DIR, num_samples=1000)
    print("Done!")
