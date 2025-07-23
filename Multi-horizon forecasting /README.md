# Multi-Horizon Forecasting for Retail Business

This project provides a solution for multi-horizon forecasting in the context of a large chain of retail stores. The main objective is to **forecast the number of daily customers up to 50 days ahead**.

This repository contains the Jupyter Notebook with the solution, along with helper scripts.

## Project Overview

This project is structured to provide a complete workflow for time series forecasting, from data exploration to model implementation and evaluation. The notebook is organized as follows:
- **Exploratory Data Analysis (EDA):** A brief analysis to understand the data's characteristics and identify potential issues.
- **Model Implementations:**
    - **Naive Model:** A benchmarking model using a weighted average of previous observations for the same day of the week.
    - **Recursive Global Gradient Boosting Model:** A recursive approach to multi-step forecasting using Gradient Boosting.
    - **Hybrid Model:** A hybrid approach that trains multiple Direct Global Gradient Boosting models for p-step predictions.

All the functions used in the notebook are available in the `utils.py` and `global_model_util.py` scripts.

## Dataset

The dataset used for this project is `n_forecast.parquet`. It's a parquet file containing transactional data for a chain of stores. The key features in the dataset include:

- `sales_date`: The date of the sales transaction.
- `store_hashed`: A unique identifier for each store.
- `n_transactions`: The number of transactions (our target variable).
- `store_format`, `zipcode_region`, `region`: Categorical features describing the store.
- Holiday and school holiday information.
- `datetime_store_open`, `datetime_store_closed`: Timestamps for store operating hours.

## Methodology

### Exploratory Data Analysis
A small EDA is performed to understand the data and detect any anomalies that might impact model performance. The analysis includes checking for missing values and identifying non-operational stores, which are then excluded from the dataset.

### Models
Three different forecasting models are implemented:

1.  **Naive Model:** This serves as a baseline and is based on a weighted average of previous observations of the same day of the week.
2.  **Recursive Global Gradient Boosting Model:** This model uses a recursive strategy, where the model's own predictions are used as input for future forecasting steps.
3.  **Hybrid Model:** This approach involves training multiple Direct Global Gradient Boosting models, each specialized for a specific prediction step `p`.

