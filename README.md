# Cluster-Based Routing in Wireless Sensor Networks using Machine Learning

## Overview
This repository contains machine learning implementations designed to optimize cluster-head selection and enhance energy efficiency in Wireless Sensor Networks (WSNs). A custom neural network model, along with baseline models such as Logistic Regression, Random Forest, and SVM, were trained and evaluated using a publicly available dataset (WSN-DS).

## Objectives
- Improve energy efficiency in WSNs through intelligent cluster-head selection.
- Compare the effectiveness of custom neural networks against traditional machine learning models.
- Demonstrate data preprocessing, model training, performance evaluation, and result visualization.

## Project Structure
- `cluster_routing_model.py`: Python script containing all preprocessing, training, evaluation, and visualization code.
- `WSN-DS.csv`: Dataset used for training and testing the models.

## Methodology
### Data Preprocessing
- Cleaned the dataset by removing irrelevant and leakage-prone features
- Performed median imputation, standard scaling, and one-hot encoding

### Machine Learning Models
- **Custom Neural Network**: Built using TensorFlow (Keras), optimized with early stopping and dropout layers
- **Baseline Models**: Logistic Regression, Random Forest, and SVM with optimized hyperparameters

### Performance Metrics
- Evaluated model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Results
- All models performed with an accuracy of 98% or greater, with the **Random Forest** model achieving the best overall performance (Recall: 95%, F1-score: 97%).
- A grouped bar chart visually compares model performance across all metrics.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage
To replicate this work:
1. Clone the repository
2. Install the required libraries (`pip install tensorflow numpy pandas scikit-learn matplotlib`)
3. Run the provided Python script

## Dataset
The dataset used in this project is publicly available: [WSN-DS](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds)

## Developer
Austin Turner
