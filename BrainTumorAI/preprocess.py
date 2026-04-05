import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "archive", "Brain Tumor Data Set", "Brain Tumor Data Set")
OUTPUT_DIR = os.path.join(BASE_DIR, "BrainTumorAI", "dataset")
CATEGORIES = ["Brain Tumor", "Healthy"]
IMG_SIZE = 224

def preprocess_images():
    data = []
    labels = []

    print(f"Data directory: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print("Data directory does not exist!")
        return

    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        
        print(f"Processing category: {category} in {path}")
        
        if not os.path.exists(path):
            print(f"Directory {path} does not exist!")
            continue

        count = 0
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                # Skip directories
                if os.path.isdir(img_path):
                    continue
                
                # Read image
                img_array = cv2.imread(img_path)
                if img_array is None:
                    continue
                
                # Resize image
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                data.append(resized_array)
                labels.append(class_num)
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} images in {category}")
                
            except Exception as e:
                print(f"Error processing {img}: {e}")

    if not data:
        print("No images found!")
        return

    print(f"Total images processed: {len(data)}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Save to directory structure
    for set_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        print(f"Saving {set_name} set...")
        for i in range(len(X)):
            label_name = "tumor" if y[i] == 0 else "no_tumor"
            target_dir = os.path.join(OUTPUT_DIR, set_name, label_name)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            file_name = f"{label_name}_{i}.jpg"
            cv2.imwrite(os.path.join(target_dir, file_name), X[i])

    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_images()
