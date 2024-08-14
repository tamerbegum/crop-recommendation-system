# Crop Recommendation System

## Overview
This project is a Crop Recommendation System that uses machine learning to suggest suitable crops based on soil conditions and climate factors. It includes data analysis, model training, and a web application for easy use.

## Features
- Data exploration and visualization of crop datasets
- Machine learning model for crop prediction
- Interactive web application for crop recommendations
- Detailed analysis of soil and climate requirements for various crops

## Technologies Used
- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for data visualization
- Scikit-learn for machine learning
- Streamlit for web application development
- Joblib for model serialization

## Project Structure
- `CropRecommendationSystem.py`: Main script for data analysis and model training
- `model_stream.py`: Script for creating and saving the machine learning pipeline
- `streamlit.py`: Streamlit web application for user interface
- `Crop_recommendation.csv`: Dataset used for training and analysis
- `stream.joblib`: Serialized machine learning model
- `/pics`: Directory containing images of various crops

## Installation
1. Clone the repository:
git clone https://github.com/yourusername/crop-recommendation-system.git
2. Install required packages:
pip install -r requirements.txt

## Usage
1. Run the Streamlit application:
streamlit run streamlit.py
2. Open the provided URL in your web browser
3. Use the interface to input soil and climate data
4. Click "RECOMMEND CROP!" to get a crop suggestion

## Data Analysis
The project includes comprehensive data analysis, featuring:
- Exploratory Data Analysis (EDA)
- Correlation analysis of soil and climate factors
- Visualization of crop distributions and requirements

## Machine Learning Model
- Utilizes Random Forest Classifier
- Includes feature engineering and preprocessing steps
- Model evaluation using various metrics (Accuracy, AUC, F1-score, etc.)

## Web Application
The Streamlit app provides:
- An intuitive interface for inputting soil and climate data
- Instant crop recommendations based on user inputs
- Visualizations of the dataset and analysis results
