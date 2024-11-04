import os
import numpy as np
import cv2

def preprocess_image(image_path, output_path):
    """Funkcja do przetwarzania obrazu: zmiana rozmiaru i zapisywanie."""
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image_array = image.astype(np.float32)
        cv2.imwrite(output_path, image)
        
        print(f"Processed and saved image at: {output_path}")
        return image_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None 

def load_data_and_save(base_dir, output_base_dir):
    """Funkcja do ładowania danych z folderu i zapisywania ich po przetworzeniu."""
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    images = []
    labels = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_base_dir, folder)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Sprawdzamy czy plik jest obrazem
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    output_path = os.path.join(output_folder_path, image_name)
                    processed_image = preprocess_image(image_path, output_path)
                    
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(folder)

    return np.array(images), np.array(labels)

# Przetwarzanie danych i zapisywanie w folderze wyjściowym
train_images, train_labels = load_data_and_save('dataset/Train', 'dataset/Train_resized')
test_images, test_labels = load_data_and_save('dataset/Test', 'dataset/Test_resized')

print(f"Loaded and processed {len(train_images)} training images and {len(test_images)} test images.")