# Stroke Prediction using Machine Learning

## Overview
This project focuses on predicting strokes using machine learning techniques. We implemented two supervised classification algorithms, Random Forest Classifier and AdaBoost Classifier, to predict whether or not a patient will experience a stroke based on a range of features such as age, BMI, glucose levels, and lifestyle factors. Additionally, we explored unsupervised learning to find natural clusters in the data.

This repository contains all the code and resources needed to reproduce the results, including data processing scripts, model training scripts, and the final report.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approaches](#modeling-approaches)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Results](#results)
- [References](#references)

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

