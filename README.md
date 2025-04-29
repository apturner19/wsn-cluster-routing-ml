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
- Performance comparison chart visualizing model effectiveness across key metrics

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage
To replicate or extend this work:
1. Clone the repository.
2. Install the required libraries (`pip install tensorflow numpy pandas scikit-learn matplotlib`)
3. Run the provided Python script to preprocess the data, train models, and visualize results

## Dataset
The dataset used in this project is the [WSN-DS](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds) dataset that is publicly available

## Developer
[Austin Turner]
