import os
import numpy as np
import cv2

def preprocess_image(image_path):
    """Funkcja do przetwarzania obrazu: zmiana rozmiaru i normalizacja."""
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image_array = image.astype(np.float32) / 255.0  # Normalizacja
        return image_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None 

def load_data(base_dir):
    """Funkcja do ładowania danych z folderu."""
    images = []
    labels = []
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path):
            for label in os.listdir(folder_path):
                label_path = os.path.join(folder_path, label)
                
                if os.path.isdir(label_path):  
                    for image_name in os.listdir(label_path):
                        print(f"Found image: {image_name}")
                        image_path = os.path.join(label_path, image_name)
                        
                        if image_path.lower().endswith((".jpg")):
                            processed_image = preprocess_image(image_path)
                            if processed_image is not None:  
                                images.append(processed_image)
                                labels.append(label)

    return np.array(images), np.array(labels)

train_images, train_labels = load_data('dataset/Training')
test_images, test_labels = load_data('dataset/Testing')

print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
