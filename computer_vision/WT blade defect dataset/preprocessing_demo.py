import cv2
import os


def resize_images(source_folder, output_folder, target_size=(1024, 1024)):
    """
    Standardizes image resolution to the target size (1024x1024) used in the WTBD dataset.

    Args:
        source_folder (str): Path to directory containing raw images.
        output_folder (str): Path to save resized images.
        target_size (tuple): Desired resolution (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(source_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Resize the image to the standard resolution
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Save the processed image
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, img_resized)
                print(f"Processed: {filename}")


if __name__ == "__main__":
    # Example usage directory structure
    # This script demonstrates the standard resizing procedure applied to the dataset.
    source_dir = "./raw_samples"
    output_dir = "./standardized_images"

    # Note: Users should replace these paths with their own data directories.
    resize_images(source_dir, output_dir)