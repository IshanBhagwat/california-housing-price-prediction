# California Housing Price Predictor

An end-to-end machine learning pipeline that predicts California median house values using Scikit-Learn. It covers data preprocessing, stratified sampling, model comparison with cross-validation, model persistence, and batch inference on new data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

## Overview

The goal is to predict `median_house_value` for districts in California based on features such as income, location, housing age, and population. The project trains multiple regression algorithms, selects the best one through cross-validation, and ships a script that is ready for production-style use:

- Trains the model once and saves it with `joblib`
- Skips retraining automatically if a saved model already exists
- Predicts on new data from `input.csv` and writes results to `output.csv`

## Dataset

- **File:** `housing.csv`
- **Target:** `median_house_value`
- **Features:** longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, ocean proximity

## Pipeline Steps

1. **Load data** from `housing.csv`.
2. **Stratified split** into train and test sets using income categories, so both sets keep the same income distribution as the full dataset.
3. **Preprocessing** with a custom Scikit-Learn pipeline:
   - Missing values handled with an imputer
   - Numerical features scaled
   - Categorical features (`ocean_proximity`) one-hot encoded
4. **Model training** with Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
5. **Evaluation** using RMSE and k-fold cross-validation.
6. **Model persistence** with `joblib`. The model is trained only if no saved file exists.
7. **Inference** on `input.csv`, with predictions saved to `output.csv`.

## Model Comparison

Random Forest gave the lowest RMSE and the most stable cross-validation results.

| Model                   | CV RMSE (mean) | Std Dev |
|-------------------------|----------------|---------|
| Linear Regression       | _add value_    | _add_   |
| Decision Tree Regressor | _add value_    | _add_   |
| **Random Forest Regressor** | **_add value_** | **_add_** |

> Replace the placeholders with the numbers from your own run.

## Project Structure

```
california-housing-price-prediction/
├── housing.csv        # training dataset
├── input.csv          # new data to predict on
├── output.csv         # generated predictions
├── main.py            # train / load model and run inference
├── model.pkl          # saved Random Forest model (created on first run)
├── requirements.txt   # Python dependencies
└── README.md
```

> Adjust file names to match your repository.

## Installation

```bash
git clone https://github.com/<your-username>/california-housing-price-prediction.git
cd california-housing-price-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

1. Put the districts you want to predict in `input.csv`. It must have the same feature columns as `housing.csv`, without `median_house_value`.
2. Run the script:

```bash
python main.py
```

- **First run:** the model is trained and saved to `model.pkl`, then predictions are generated.
- **Later runs:** the saved model is loaded directly, which is much faster.
3. Open `output.csv` to see the predicted `median_house_value` for each row.

To force retraining, delete `model.pkl` and run the script again.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- Joblib
- Matplotlib

## Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Try Gradient Boosting or XGBoost
- Feature engineering (rooms per household, population per household)
- Wrap the model in a REST API with FastAPI or Flask
- Add a simple web UI for single predictions

## License

This project is licensed under the MIT License.
