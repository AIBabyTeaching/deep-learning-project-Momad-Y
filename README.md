# Deep Learning - Celebrity Look-Alike Project

## Project Description

The aim of this project is to develop a deep learning system for accurately matching facial features with those of celebrities. The ultimate goal is the deployment for user-based applications. In the field of computer vision, this project is crucial as it demonstrates the power of Convolutional Neural Networks (CNNs) in facial feature extraction, pattern recognition, and their practical applications in entertainment and human-computer interaction. The project tackles various computer vision tasks, including Facial Recognition, Feature Extraction, and Similarity Scoring, with the primary objective of identifying celebrities resembling input individuals.

## Problem Statement

The project aims to build a Celebrity Look-Alike Model using Convolutional Neural Networks (CNN), tapping into the demand for tech-driven entertainment. Users can enjoy a lighthearted exploration of their celebrity doppelgängers. The project integrates image processing, CNN implementation, and practical neural network applications, aligning with the course's focus on the practical implementation of Deep Neural Networks.

## Methodology (Tech Stack)

In this project, Python and Jupyter Notebook will serve as the primary programming languages. TensorFlow and Keras will be employed as the main framework for constructing the model. Additionally, for dataset preprocessing and visualization, we will utilize NumPy, Pandas, and Matplotlib, and for computer vision techniques, CV2 will be used. Flask will be employed as the backend for web deployment, while React will serve as the front-end framework.

Python and Jupyter Notebook are preferred for their conducive environment for code development and documentation. TensorFlow and Keras are renowned frameworks in the field of deep learning. NumPy, Pandas, and Matplotlib play crucial roles in the preprocessing and visualization of datasets. Additionally, CV2 will be employed for tasks such as image loading, preprocessing, face detection, and visualization.

## Data-Set

The dataset employed in this project is provided by ETH Zurich University. Originally shared by researchers for age and gender prediction tasks, the IMDB Faces dataset will be exclusively utilized. The dataset is structured in the MATLAB file format (.mat) and encompasses ten distinct features, including date of birth, year the photo was taken, image path, gender, celebrity name, face location, face score, second face score, celebrity names, and celebrity ID. The dataset comprises approximately 460,000 records, with an overall size of approximately 6.5 GB.

## Implementation

### 1. Dataset Validation

-   #### 1.1 Statistical Analysis
    -   1.1.1 Correlations
    -   1.1.2 Data describe
-   #### 1.2 Data Cleaning:
    -   1.2.1 Insert imputers and Encode if required
    -   1.2.2 Drop unnecessary columns

### 2. Visualization:

-   #### 2.1 Exploratory Data Analysis
    -   Univariate Exploration
    -   Bivariate Exploration
    -   Multivariate Exploration

### 3. Preprocessing:

-   #### 3.1 Feature normalization
    -   3.1.1 Standard Scaling
-   #### 3.2 Regularization techniques
    -   3.2.1 Dropout
    -   3.2.2 L1 and L2 Types

### 4. Neural Network construction

-   #### 4.1 Keras Model creation (Fully Connected NN/Convolutional NN/Recurrent NN)
-   #### 4.2 Keras Compiler
-   #### 4.3 Model fitting
-   #### 4.4 Model predicting

### 5. Evaluation metrics:

-   #### 5.1 Training Accuracy
-   #### 5.2 Training Loss
-   #### 5.3 Test Accuracy
-   #### 5.4 Test Loss

### 6. Documentation of insights in Jupyter notebooks.
