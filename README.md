🐱🐶 Cat vs Dog Image Classification using CNN
 
📌 Overview
This project focuses on classifying images of cats and dogs using Convolutional Neural Networks (CNNs). It is a deep learning-based project that employs a supervised learning approach to achieve high accuracy in image classification.

🚀 Features
Uses a CNN model to classify images into two categories: Cats and Dogs.
Trained on a dataset of labeled cat and dog images.
Utilizes TensorFlow/Keras for deep learning implementation.
Achieves high accuracy through data augmentation and model tuning.
Supports model evaluation using accuracy, loss curves, and confusion matrices.

📂 Dataset
The dataset used is the Dogs vs. Cats dataset from Kaggle, which contains:

12,500 cat images 🐱
12,500 dog images 🐶
Balanced dataset with equal classes.
🔗 Dataset Source: Kaggle - Dogs vs Cats

📖 Project Structure
📦 Cat-vs-Dog-Classification
 ┣ 📂 dataset
 ┃ ┣ 📂 train
 ┃ ┣ 📂 test
 ┣ 📂 models
 ┃ ┣ best_model.h5
 ┣ 📂 notebooks
 ┃ ┣ cat_vs_dog_classification.ipynb
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📜 train.py
 ┣ 📜 test.py
 ┗ 📜 predict.py

🏗️ Model Architecture
The CNN model consists of:
--> Convolutional Layers (Conv2D): Extracts features from images.
--> Pooling Layers (MaxPooling2D): Reduces feature map size.
--> Flatten Layer: Converts 2D features to a 1D array.
--> Dense (Fully Connected) Layers: Classifies images into cat or dog.
--> Activation Functions: Uses ReLU and Softmax for non-linearity and probability output.

📊 Model Evaluation
Accuracy: 96%
Loss Curve & Accuracy Plot: Available in the Google Colab.
Confusion Matrix: Shows model performance on test data.
