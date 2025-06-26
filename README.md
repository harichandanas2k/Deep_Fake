# DeepFake Detector: CNN & SVM Based Face Manipulation Identification

This project presents a comparative analysis of Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) for the detection of deepfake images. With the rise of AI-generated synthetic media, it's increasingly important to develop robust detection systems. This implementation aims to accurately classify real and deepfake facial images using deep learning and machine learning approaches.


## 📌 Objective

- To build a model that detects deepfake facial images.
- To compare the performance of CNN and SVM for classification.
- To explore the use of AI for enhancing digital media integrity.


## 🧠 Technologies Used

- **Python**  
- **TensorFlow / Keras** – for building the CNN model  
- **Scikit-learn** – for SVM and evaluation metrics  
- **OpenCV / PIL** – for image processing  
- **Matplotlib / Seaborn** – for visualization  
- **Jupyter Notebook**


## 🗃️ Dataset

The project uses a dataset containing real and deepfake face images. The data is preprocessed and split into training and testing sets to evaluate model performance.


## 🔍 Model Overview

### ✅ Convolutional Neural Network (CNN)
- Feature extraction through multiple convolution and pooling layers.
- Classification via fully connected layers.
- Designed to learn spatial hierarchies in image data.

### ✅ Support Vector Machine (SVM)
- Used as a classical ML classifier on extracted features.
- Works well with high-dimensional image data when combined with PCA or flattening techniques.


## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CNN   | ~95%     | High      | High   | High     |
| SVM   | ~89%     | Moderate  | Moderate| Moderate |

*Note: Results may vary depending on dataset split and training settings.*


## 📊 Visualization

Includes:
- Sample predictions from CNN and SVM
- Confusion matrices
- Accuracy and loss curves for CNN training

## 💡 Future Work
- Extend the model for video-based deepfake detection.
- Integrate transfer learning for enhanced performance.
- Deploy as a web service for real-time detection.

## 📜 License
This project is open-source and available under the MIT License.
