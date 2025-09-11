import os
import cv2
import numpy as np
from tqdm import tqdm

def load_train_data(folder_path):
    """
    Loads training images from a folder containing subfolders for each class (A, B, nothing, etc.).
    """
    images = []
    labels = []

    if not os.path.isdir(folder_path):
        print(f"Error: Dataset folder '{folder_path}' not found.")
        return None, None

    print(f"\nScanning training images in '{os.path.basename(folder_path)}'...")
    
    all_image_paths = []
    for label_folder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, label_folder)
        if not os.path.isdir(subfolder_path):
            continue
        for image_filename in os.listdir(subfolder_path):
            full_path = os.path.join(subfolder_path, image_filename)
            all_image_paths.append((full_path, label_folder))

    for image_path, label in tqdm(all_image_paths, desc=f"Processing {os.path.basename(folder_path)}"):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"Found {len(images)} training images for {len(np.unique(labels))} classes.")
    return np.array(images), np.array(labels)

def load_test_data_flat(folder_path):
    """
    Loads test images from a single 'flat' folder.
    Assumes the label is the prefix of the filename (e.g., 'A_1.jpg', 'space_test.jpg').
    """
    images = []
    labels = []

    if not os.path.isdir(folder_path):
        print(f"Error: Dataset folder '{folder_path}' not found.")
        return None, None

    print(f"\nScanning test images in '{os.path.basename(folder_path)}'...")
    
    image_filenames = os.listdir(folder_path)
    for image_filename in tqdm(image_filenames, desc=f"Processing {os.path.basename(folder_path)}"):
        image_path = os.path.join(folder_path, image_filename)
        try:
            # Extract label from filename, assuming format 'label_....jpg'
            label = os.path.basename(image_filename).split('_')[0]
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            
    print(f"Found {len(images)} test images for {len(np.unique(labels))} classes.")
    return np.array(images), np.array(labels)


if __name__ == '__main__':
    # ==================================================================
    # IMPORTANT: EDIT THESE PATHS
    TRAIN_DATA_PATH = r"D:/LTIMintree_project/ASL_Alphabet_Dataset/asl_alphabet_train"
    TEST_DATA_PATH = r"D:/LTIMintree_project/ASL_Alphabet_Dataset/asl_alphabet_test_augmented"
    # ==================================================================

    # Load the training data using the folder-based loader
    X_train, y_train = load_train_data(TRAIN_DATA_PATH)
    
    # Load the testing data using the new flat-file loader
    X_test, y_test = load_test_data_flat(TEST_DATA_PATH)

    if X_train is not None and y_test is not None: # Check y_test as X_test could be empty
        print("\nSaving data to .npy files...")
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)

        print("\n--- Final Data Shapes ---")
        print(f"Training data shape (X_train): {X_train.shape}")
        print(f"Testing data shape (X_test): {X_test.shape}")
        print("\n✅ Preprocessing complete. Data saved to .npy files.")
    else:
        print("\n❌ Preprocessing failed. Please check the folder paths and their contents.")