# Stroke Prediction using Machine Learning

## Overview
This project focuses on predicting strokes using machine learning techniques. We implemented two supervised classification algorithms, Random Forest Classifier and AdaBoost Classifier, to predict whether or not a patient will experience a stroke based on a range of features such as age, BMI, glucose levels, and lifestyle factors. Additionally, we explored unsupervised learning to find natural clusters in the data.

This repository contains all the code and resources needed to reproduce the results, including data processing scripts, model training scripts, and the final report.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approaches](#modeling-approaches)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Results](#results)
- [References](#references)

## Project Structure

The repository is organized as follows:

- **data/**: This folder contains the dataset files used in this project. The dataset is preprocessed and used for training and testing the machine learning models.

- **scripts/**: This folder contains the Python scripts used for classification tasks.
  - `supervised_classification.py`: Script for implementing supervised classification using Random Forest and AdaBoost algorithms.
  - `unsupervised_classification.py`: Script for exploring unsupervised learning techniques.
## Dataset
The dataset contains 5110 patient records with 12 features related to demographic information, health history, and lifestyle choices. The dataset was sourced from Kaggle and does not have any associated publication.

- **Size**: 5110 records
- **Imbalance**: The dataset is highly imbalanced, with only 250 positive stroke cases and 4860 negative cases.

### Attributes
The dataset includes the following features:
- `id`: Unique identifier
- `gender`: Male, Female, or Other
- `age`: Age of the patient
- `hypertension`: Whether the patient has hypertension (0: No, 1: Yes)
- `heart_disease`: Whether the patient has heart disease (0: No, 1: Yes)
- `ever_married`: Whether the patient has ever been married (Yes or No)
- `work_type`: Type of employment (children, Govt_job, Never_worked, Private, Self-employed)
- `Residence_type`: Rural or Urban
- `avg_glucose_level`: Average glucose level in blood (mg/dL)
- `bmi`: Body Mass Index
- `smoking_status`: Formerly smoked, Never smoked, Smokes, or Unknown
- `stroke`: Whether the patient has had a stroke (0: No, 1: Yes)

## Installation
To run this project, you will need to have the following installed:
- Python 3.8+
- `pip` for package management

### Dependencies
Install all dependencies using the following command:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/DarshilGajjar/Stroke-Prediction-using-ML.git
    cd Stroke_Prediction_ML
    ```

2. Prepare the dataset:
    - Place the dataset (`stroke_data.csv`) in the `data/` directory.

3. Run the Jupyter notebooks:
    - You can view and modify the exploratory data analysis and model training process using the notebooks in the `notebooks/` folder.

4. Run the Python scripts:
    - **Supervised Classification**:
      ```bash
      python srcipts/supervised_classification.py
      ```
    - **Unsupervised Classification**:
      ```bash
      python srcipts/unsupervised_classification.py
      ```

## Modeling Approach
The modeling approach involves both supervised and unsupervised learning techniques to predict strokes based on various patient features.

### Supervised Learning
1. **Classifiers Used**:
   - **Random Forest Classifier**: An ensemble learning method that operates by constructing multiple decision trees during training and outputs the mode of the classes.
   - **AdaBoost Classifier**: An adaptive boosting technique that combines multiple weak classifiers to create a strong classifier.

2. **Data Splitting**:
   - The dataset is split into training (approximately 75%) and testing (approximately 25%) sets. For this study, we used 449 patient data for training and 150 patient data for testing.

3. **Performance Evaluation**:
   - Accuracy and Area Under the Curve (AUC) metrics are calculated to evaluate model performance:
     - Accuracy of Random Forest Classifier: 75.33%
     - Accuracy of AdaBoost Classifier: 70.67%
     - AUC for Random Forest Classifier: 0.75
     - AUC for AdaBoost Classifier: 0.69

### Unsupervised Learning
1. **Clustering**:
   - KMeans clustering is used to identify patterns in the data based on three selected features: Age, BMI, and Average Glucose Level.

2. **Visualization**:
   - 1D histograms and 3D scatter plots are generated to visualize the distribution of data points and the separability of clusters.

3. **Cluster Evaluation**:
   - Silhouette and Calinski-Harabasz indexes are calculated to assess the clustering quality, indicating well-distributed data and distinguishable clusters.

## Results

### Supervised Classification Results
1. **Random Forest Classifier**:
   - **Accuracy**: 75.33%
   - **AUC (Area Under the Curve)**: 0.75
   - The Random Forest classifier demonstrated strong performance in classifying stroke occurrences, likely due to its ability to handle the imbalanced dataset effectively.

2. **AdaBoost Classifier**:
   - **Accuracy**: 70.67%
   - **AUC**: 0.69
   - AdaBoost performed slightly worse compared to Random Forest but still achieved reasonable accuracy given the challenging nature of stroke prediction with imbalanced data.

### Unsupervised Classification Results
1. **KMeans Clustering**:
   - The KMeans algorithm identified two clear clusters based on the features: Age, BMI, and Average Glucose Level. These clusters are somewhat separable, indicating the presence of two potential classes: stroke and non-stroke patients.
   - **Silhouette Index**: Shows that the clusters are reasonably well-defined.
   - **Calinski-Harabasz Index**: Confirms that the clusters are distinct and well-separated in the feature space.

2. **Cluster Visualization**:
   - **3D Scatter Plot**: Demonstrates a clear division between the two clusters for stroke and non-stroke patients.
   - **1D Histograms**: Show the distribution of the three key features (Age, BMI, Average Glucose Level) for the two classes.

### Key Observations:
- **Imbalanced Dataset**: The original dataset was highly imbalanced, with only 250 stroke cases out of 5110 observations. After balancing the dataset, both classifiers showed improved performance.
- **Model Performance**: Random Forest outperformed AdaBoost, likely due to its ability to manage imbalanced data more effectively.
- **Clustering**: KMeans clustering indicated two well-defined classes, providing further validation of the supervised classification results.

## References

1. **Heo J, Yoon JG, Park H, Kim YD, Nam HS, Heo JH**. 
   *Machine Learning-Based Model for Prediction of Outcomes in Acute Stroke.* Stroke. 2019 May;50(5):1263-1265. 
   - DOI: [10.1161/STROKEAHA.118.024293](https://doi.org/10.1161/STROKEAHA.118.024293)
   - PMID: 30890116.

2. **Alanazi EM, Abdou A, Luo J**. 
   *Predicting Risk of Stroke From Lab Tests Using Machine Learning Algorithms: Development and Evaluation of Prediction Models.* JMIR Form Res. 2021 Dec 2;5(12):e23440. 
   - DOI: [10.2196/23440](https://doi.org/10.2196/23440)
   - PMID: 34860663; PMCID: PMC8686476.

3. **Dritsas E, Trigka M**. 
   *Stroke Risk Prediction with Machine Learning Techniques.* Sensors (Basel). 2022 Jun 21;22(13):4670. 
   - DOI: [10.3390/s22134670](https://doi.org/10.3390/s22134670)
   - PMID: 35808172; PMCID: PMC9268898.