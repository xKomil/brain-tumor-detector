# Brain Tumor Detector

## Overview
The **Brain Tumor Detector** is a machine learning application that utilizes deep learning techniques to classify brain MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. This project leverages the MobileNetV2 convolutional neural network architecture to analyze images and provide accurate predictions based on the trained dataset.

## Dataset
The dataset is sourced from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It contains MRI images categorized into the following classes:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The dataset is organized into separate directories for training and testing, facilitating the implementation of effective data loading and preprocessing routines.

## Project Components

### 1. Data Preprocessing
- Images are resized to 128x128 pixels and normalized to a pixel value range of [0, 1].
- Data augmentation techniques, such as rotation, shifting, zooming, and flipping, are applied to the training dataset to enhance the model's robustness and prevent overfitting.

### 2. Model Architecture
- The MobileNetV2 architecture is used as the base model, pre-trained on the ImageNet dataset. This transfer learning approach allows the model to leverage learned features.
- Custom layers, including global average pooling and a fully connected layer with a softmax activation function, are added to adapt the model for the classification task.

### 3. Training and Validation
- The model is compiled with the Adam optimizer and categorical crossentropy loss function, suitable for multi-class classification.
- The training process involves:
  - Initially freezing the base model layers to train only the newly added layers.
  - Subsequently unfreezing the base model layers to fine-tune the entire model.
- The model achieves a test accuracy of **80.62%**, demonstrating effective performance in classifying brain tumors.

### 4. Deployment
- A web interface is developed using **Streamlit**, allowing users to upload MRI scans for classification. The model processes the uploaded image and outputs the predicted class, providing an intuitive experience for users.

### 5. Results Interpretation
- If the model predicts "no tumor," a specific message is displayed to inform the user. For other classifications, the corresponding tumor type is indicated.

## Conclusion
The Brain Tumor Detector project demonstrates the application of deep learning in medical diagnostics, specifically in analyzing MRI images for tumor classification. While the current accuracy stands at 80.62%, there is potential for improvement through further data augmentation, hyperparameter tuning, and exploring alternative model architectures.

## Installation and Usage
To set up and run the project locally, follow these steps:

1. **Clone the repository:**
   git clone <repository-url>
   cd brain-tumor-detector

2. **Install the required packages:**
pip install -r requirements.txt

3. ****Run the Streamlit app:**
streamlit run app.py

