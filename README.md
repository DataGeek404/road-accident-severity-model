# Road Accident Severity Prediction Model

This repository contains a machine learning model that predicts road accident severity using various factors such as age, gender, driving experience, and type of vehicle. The model is built using **Python**, **scikit-learn**, and other essential data preprocessing tools.

## Project Overview

Road accidents are a critical issue, and predicting the severity of accidents can help in resource allocation and improving safety measures. This project aims to predict accident severity based on a dataset of historical road accident data.

### Key Features
- **Data Preprocessing**: Handling missing values and categorical feature encoding.
- **Modeling**: Uses Linear Regression for predicting accident severity.
- **Evaluation**: Mean Squared Error (MSE) is used to evaluate the model’s performance.
  
## Dataset

The dataset used for this project contains information about road accidents, including:
- `Age_band_of_driver`
- `Sex_of_driver`
- `Educational_level`
- `Driving_experience`
- `Type_of_vehicle`
- `Accident_severity` (Target Variable)

The dataset is cleaned and preprocessed to handle missing values and categorical features.

## Model Pipeline

The data pipeline is set up as follows:
1. **Preprocessing**: Categorical variables are one-hot encoded.
2. **Model Training**: Linear Regression is applied to predict accident severity.
3. **Evaluation**: The model’s performance is measured using the Mean Squared Error (MSE).

## Installation

To run the project locally, follow these steps:

### Prerequisites

- Python 3.x
- Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
