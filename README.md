# AI-Based-Crop-Disease-Detection-System

### Overview

This project is a deep learning-based web application for automated crop disease classification using plant leaf images. The system uses a fine-tuned ResNet50 CNN model trained on the PlantVillage dataset containing over 54,000 labeled images across 15 classes.

The application allows users to upload leaf images and receive real-time disease predictions along with Grad-CAM visualizations for model interpretability.

##Features

Home Page: Upload plant leaf images for disease classification.
Prediction Page: Analyze uploaded images using the trained ResNet50 model.
Results Page: View uploaded image, predicted disease name, confidence score, and Grad-CAM heatmap visualization.
Download Feature: Save the generated Grad-CAM heatmap image for future reference.

##Model Performance

Test Accuracy: 97%
Average Inference Time: ~200 ms per image
Dataset: PlantVillage Dataset
Number of Classes: 15

##Requirements

Python 3.8+
Flask
OpenCV
Matplotlib
Machine Learning Libraries: scikit-learn, XGBoost, TensorFlow, pandas, numpy
 
##Install dependencies:

pip install -r requirements.txt  

##Run the Flask application:

python app.py 
 
##Open the application in a browser:

http://127.0.0.1:5000

##File Structure

app.py: Main Flask application.
templates/: HTML files for the UI.
static/: CSS/JS assets.
models/: Pre-trained machine learning models.
data/: Sample datasets for testing.
