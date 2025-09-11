import cv2
import numpy as np
import os
from tqdm import tqdm

def augment_image(image):
    """Applies a set of simple augmentations to a single image."""
    augmented_images = []
    
    # 1. Original Image
    augmented_images.append(image)
    
    # 2. Rotations
    rows, cols, _ = image.shape
    M_rotate_pos = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)  # Rotate 5 degrees
    M_rotate_neg = cv2.getRotationMatrix2D((cols / 2, rows / 2), -5, 1) # Rotate -5 degrees
    augmented_images.append(cv2.warpAffine(image, M_rotate_pos, (cols, rows)))
    augmented_images.append(cv2.warpAffine(image, M_rotate_neg, (cols, rows)))

    # 3. Brightness changes
    bright = np.ones(image.shape, dtype="uint8") * 50
    dark = np.ones(image.shape, dtype="uint8") * 50
    augmented_images.append(cv2.add(image, bright))
    augmented_images.append(cv2.subtract(image, dark))
    
    # 4. Adding Noise
    noise = np.random.randint(0, 25, image.shape, dtype='uint8')
    augmented_images.append(cv2.add(image, noise))
    
    # 5. Flipping (if applicable, be careful with letters that are not symmetrical)
    # augmented_images.append(cv2.flip(image, 1))

    # Add more augmentations like zoom, shear, etc. if needed to reach the target count
    while len(augmented_images) < 10: # Ensure we have at least 10 images
         augmented_images.append(image) # Just duplicate if other methods are not enough

    return augmented_images[:10] # Return exactly 10 images


if __name__ == '__main__':
    # ==================================================================
    # IMPORTANT: EDIT THESE PATHS
    INPUT_TEST_FOLDER = r"D:/LTIMintree_project/ASL_Alphabet_Dataset/asl_alphabet_test"
    OUTPUT_TEST_FOLDER = r"D:/LTIMintree_project/ASL_Alphabet_Dataset/asl_alphabet_test_augmented"
    # ==================================================================

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_TEST_FOLDER):
        os.makedirs(OUTPUT_TEST_FOLDER)
        print(f"Created output folder: {OUTPUT_TEST_FOLDER}")

    image_files = os.listdir(INPUT_TEST_FOLDER)

    print(f"Found {len(image_files)} images to augment...")

    for image_filename in tqdm(image_files, desc="Augmenting Images"):
        image_path = os.path.join(INPUT_TEST_FOLDER, image_filename)
        
        # Read the original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue
            
        # Get the label (e.g., 'A' from 'A_test.jpg')
        label = image_filename.split('_')[0]
        
        # Create augmented versions
        augmented_images = augment_image(original_image)
        
        # Save the new images
        for i, aug_image in enumerate(augmented_images):
            new_filename = f"{label}_{i}.jpg"
            save_path = os.path.join(OUTPUT_TEST_FOLDER, new_filename)
            cv2.imwrite(save_path, aug_image)

    print("\n✅ Augmentation complete!")
    print(f"New test images are saved in: {OUTPUT_TEST_FOLDER}")