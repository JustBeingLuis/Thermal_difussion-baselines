import argparse
import os
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Save images from Flowers dataset for FID evaluation.")
    parser.add_argument("--n_images", type=int, default=10, help="Number of images per class to save.")
    parser.add_argument("--output_dir", type=str, default="true_flowers_test", help="Output directory where images will be saved.")
    parser.add_argument("--image_size", type=int, default=128, help="Image size (resize and center crop).")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train, test, val).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Transformation pipeline matching the training preprocessing (geometrically)
    # Resize keeping aspect ratio, then center crop to square
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
    ])

    print(f"Loading dataset split '{args.split}'...")
    # Load the dataset
    dataset = load_dataset("nelorth/oxford-flowers", split=args.split, trust_remote_code=True)

    class_counts = {}
    total_saved = 0
    
    print(f"Processing images (Goal: {args.n_images} per class)...")
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        label = item['label']
        
        # Initialize counter for this class if not exists
        if label not in class_counts:
            class_counts[label] = 0
            
        # Check if we need more images for this class
        if class_counts[label] < args.n_images:
            image = item['image'] # Expecting PIL Image
            
            # Apply transforms
            processed_image = transform(image)
            
            # Save image
            # Naming format: class_count.png to keep it unique per class and simple
            # Or unique global name
            filename = f"class_{label}_img_{class_counts[label]}.png"
            processed_image.save(os.path.join(args.output_dir, filename))
            
            class_counts[label] += 1
            total_saved += 1
    
    print(f"Done. Saved a total of {total_saved} images to '{args.output_dir}'.")
    
    # Report on classes that didn't meet the target
    incomplete_classes = [k for k, v in class_counts.items() if v < args.n_images]
    if incomplete_classes:
        print(f"Warning: The following classes have fewer than {args.n_images} images in the '{args.split}' split: {incomplete_classes}")

if __name__ == "__main__":
    main()
