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

## How to Run

### Clone this repository:

```bash
git clone https://github.com/DataGeek404/accident-severity-prediction.git
cd accident-severity-prediction


# Road Accident Severity Prediction

This project aims to predict the severity of road accidents using machine learning techniques.

## Run the Model

The Jupyter notebook provided in the repository contains all the steps to preprocess the data, train the model, and make predictions.

```bash
jupyter notebook road_accident_severity_model.ipynb

## Load the Trained Model

You can load the pre-trained model saved as `road_accident_severity_model.pkl` and make predictions with new data:

```python
import joblib

# Load the saved model
model = joblib.load('road_accident_severity_model.pkl')

# Make predictions with new data
predictions = model.predict(new_data)


## Files in the Repository

- **road_accident_severity_model.pkl**: The trained model.
- **road_accident_severity_model.ipynb**: Jupyter notebook with data preprocessing, model training, and evaluation.
- **README.md**: Project overview and instructions.
- **requirements.txt**: Required dependencies to run the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or suggestions, feel free to contact me:

- GitHub: [DataGeek404](https://github.com/DataGeek404)

