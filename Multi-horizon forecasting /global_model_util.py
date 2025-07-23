import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from collections import deque
import utils


def objective_function(trial, X_data, y_data, categorical_feature_names):
    """
    Objective function for Optuna to minimize.
    Performs time-series cross-validation.
    Input:
    - trial: Optuna trial object
    - X_data: DataFrame with the training data
    - y_data: Series with the target variable
    - categorical_feature_names: list of categorical features
    Output:
    - mean_squared_error: mean squared error of the model

    """
    # 1. Define Hyperparameter Search Space for this trial
    params = {
        'objective': 'poisson',  # For MAE
        'metric': 'poisson',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 91,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 300, 3000, step=100), # Set high, early stopping will handle
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # 2. Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3, )
    
    fold_scores = []
    actual_categorical_features_in_X = [col for col in categorical_feature_names if col in X_data.columns]


    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric=params['metric'], 
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                  categorical_feature=actual_categorical_features_in_X
                 )
        
        preds = model.predict(X_val_fold)
        score = mean_squared_error(y_val_fold, preds) 
        fold_scores.append(score)
        

    return np.mean(fold_scores) 


def get_buffer_df(df_train, size = 50):
    '''
    Generate a buffer dataframe for the rolling forecast.
    The buffer dataframe contains the last 49 days of the training set.
    Input:
    - df_train: DataFrame with the training data
    - size: number of days to include in the buffer
    Output:
    - buffer_df: DataFrame with the buffer data
    '''
    def gen_buffer_store(df_train, store): 
        '''
        Generate a buffer dataframe for the rolling forecast.
        The buffer dataframe contains the last 49 days of the training set.
        '''
        
        store_df = df_train[df_train['store_hashed'] == store]
        if store_df.shape[0] > size:
            store_df = store_df.sort_values('sales_date')
            buffer_df = store_df.iloc[-50:]
            return buffer_df
        return store_df
    
    buffer_df = pd.DataFrame()
    for store in df_train['store_hashed'].unique():
        store_buffer_df = gen_buffer_store(df_train, store)
        buffer_df = pd.concat([buffer_df, store_buffer_df], ignore_index=True)
    return buffer_df

def recursive_forecast(buffer_df, test_set,
                        model, 
                        lags,
                        categorical_features = ['store_hashed', 'store_format', 'zipcode_region', 'region', 'openning_hour', 
                                            'closing_hour', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month'], 
                        horizon = 50): 
    '''
    Perform the recursive forecast. The function uses the last 50 days of the training set as a buffer data set
    to left create the lagged variables.
    The buffer is updated with the predictions of the model.
    Input:
    - buffer_df: DataFrame with enough data to create the lagged variables
    - test_set: DataFrame with the test set
    - model: model to use for the forecast
    - lags: list of lags to use for the model
    - categorical_features: list of categorical features
    - horizon: number of days to forecast
    Output:
    - predictions_df: DataFrame with the predictions
    '''
    df_test = test_set.copy()

    #make sure that the predictions are made for the 50 days horizon
    df_test['relative_horizon'] = df_test.groupby('store_hashed', observed = True).cumcount() + 1
    maximum_horizon = df_test['relative_horizon'].unique()[horizon-1] 
    df_test = df_test[df_test['relative_horizon'] <= maximum_horizon].copy()
    for col in df_test.columns:
        if col in ['is_national_holiday', 'is_test', 'relative_horizon']:
            df_test = df_test.drop(columns=[col])
   
    df_test = df_test.sort_values(['store_hashed', 'sales_date'])
    buffer_df = buffer_df.sort_values(['store_hashed', 'sales_date'])

    all_predictions_records = []
    predictive_features = model.feature_name_
    for store in df_test['store_hashed'].unique(): 
        store_df_test = df_test[df_test['store_hashed'] == store].copy()
        store_df_buffer = buffer_df[buffer_df['store_hashed'] == store].copy()
        #transaction_values = deque(store_df_buffer['n_transactions'].values)
        dates_in_buffer = deque(store_df_buffer['sales_date'].values)
        dates_predicted = dict()

        for i in range(store_df_test.shape[0]):

            row = store_df_test.iloc[i].copy()
            for lag in lags:
                look_up_date = row['sales_date'] - pd.DateOffset(lag, 'D')
                #check if the date is in the buffer
                if look_up_date in dates_in_buffer:
                    row[f'n_transactions_lag_{lag}'] = store_df_buffer[store_df_buffer['sales_date'] == pd.to_datetime(look_up_date)]['n_transactions'].values[0]
                elif look_up_date in dates_predicted.keys():
                    row[f'n_transactions_lag_{lag}'] = dates_predicted[look_up_date]
                else:
                    row[f'n_transactions_lag_{lag}'] = pd.NA 


                #if lag < len(transaction_values):
                #    row[f'n_transactions_lag_{lag}'] = transaction_values[-lag]
                #else: 
                #    row[f'n_transactions_lag_{lag}'] = pd.NA
                
            #ensure that the row is in the same format as the training set
            
            features = row[predictive_features].to_frame().T
            #make sure that categorical features are in the same format as the training set
            for col in predictive_features:
                if col in categorical_features:
                    features[col] = features[col].astype('category')
                else: 
                    features[col] = pd.to_numeric(features[col], errors='coerce')
            #predict the number of transactions
            prediction = model.predict(features)
            #update the buffer dataframe with the new row
            dates_predicted[row['sales_date']] = prediction[0]
            all_predictions_records.append({
                        'store_hashed': store,
                        'sales_date': row['sales_date'],
                        'predicted_n_transactions': prediction[0],
                        'relative_horizon': i + 1 # Assuming 50 days are processed for each store
                    })
    predictions_df = pd.DataFrame(all_predictions_records)
    final_predictions_df = pd.merge(
            predictions_df,
            df_test[['store_hashed', 'sales_date', 'n_transactions']],
            on=['sales_date', 'store_hashed'],
            how='left' # Use left merge to keep all original test rows
        )            
    
    
    return final_predictions_df

import os
import joblib

def train_direct_models(df, 
                         categorical_features, 
                        n_direct_models = 7, 
                        objective_function = objective_function,
                        lags = [1, 2, 3, 4, 5, 6, 7,8, 14, 21, 28, 35, 42, 49],
                        ):
    """
    Train n_direct_models LightGBM models for each lagged target.
    The models are saved in the directory "my_direct_models".
    Input: 
    - df: DataFrame with the training data
    - categorical_features: list of categorical features
    - n_direct_models: number of direct models to train
    - objective_function: objective function for Optuna
    - lags: list of lags to use for the models

    """
    models_save_directory = "my_direct_models"
    os.makedirs(models_save_directory, exist_ok=True)

    models = {}
    for col in ['datetime_store_open', 'datetime_store_closed', 'is_national_holiday', 'is_test']: 
        if col in df.columns: 
            df = df.drop(columns=[col])
    
    for i in range(2, n_direct_models+1): 
        df_train = utils.add_lags(df, np.array(lags)+i-1)
        X_train = df_train.drop(columns=['n_transactions', 'sales_date'])
        y_train = df_train['n_transactions']

        for col in categorical_features:
            if col in X_train.columns: 
                X_train[col] = X_train[col].astype('category')

        #Perform hyperparameter tuning using Optuna        
        study = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)) 

        
        study.optimize(lambda trial: objective_function(trial, X_train, y_train, categorical_features), 
                        n_trials=20) # Adjust n_trials based on available time/resources

        print("\nHyperparameter tuning complete.")
        best_trial = study.best_trial
        best_lgbm_params = best_trial.params
        model = lgb.LGBMRegressor(**best_lgbm_params)
        model.fit(X_train, y_train,
                  categorical_feature=[col for col in categorical_features if col in X_train.columns])
        models[f'model_{i}'] = model
        print(f"Model {i} trained.")
    for model_name, model_instance in models.items():
        file_path = os.path.join(models_save_directory, f"{model_name}.joblib")
        joblib.dump(model_instance, file_path)
        print(f"Model {model_name} saved to {file_path}")

def hybrid_forecast(buffer_df, 
                    test_set, 
                    models, 
                    categorical_features,
                    lags = [1, 2, 3, 4, 5, 6, 7,8, 14, 21, 28, 35, 42, 49],
                    horizon = 50):
    df_test = test_set.copy()
    '''
    Perform the hybrid forecast. The function uses the last 50 days of the training set as a buffer data set
    to left create the lagged variables.
    The buffer is updated with the predictions of the model.
    Input: 
    - buffer_df: DataFrame with enough data to create the lagged variables
    - test_set: DataFrame with the test set
    - models: dictionary of models to use for the forecast
    - categorical_features: list of categorical features
    - lags: list of lags to use for the models
    - horizon: number of days to forecast
    Output:
    - predictions_df: DataFrame with the predictions
    '''

    #make sure that the predictions are made for the 50 days horizon
    if 'relative_horizon' not in df_test.columns:
        df_test['relative_horizon'] = df_test.groupby('store_hashed', observed = True).cumcount() + 1
    maximum_horizon = df_test['relative_horizon'].unique()[horizon-1] 
    df_test = df_test[df_test['relative_horizon'] <= maximum_horizon].copy()
    for col in df_test.columns:
        if col in ['is_national_holiday', 'is_test', 'relative_horizon']:
            df_test = df_test.drop(columns=[col])
   
    df_test = df_test.sort_values(['store_hashed', 'sales_date'])
    buffer_df = buffer_df.sort_values(['store_hashed', 'sales_date'])

    all_predictions_records = []
    for store in df_test['store_hashed'].unique(): 
        store_df_test = df_test[df_test['store_hashed'] == store].copy()
        store_df_buffer = buffer_df[buffer_df['store_hashed'] == store].copy()
        #transaction_values = deque(store_df_buffer['n_transactions'].values)
        dates_in_buffer = deque(store_df_buffer['sales_date'].values)
        dates_predicted = dict()
        for i in range(1, store_df_test.shape[0]+1):
            #choose model to use
            model_index = (i-1) % len(models)
            model = models[f'model_{model_index+1}']
            row = store_df_test.iloc[i-1].copy()
            for lag in np.array(lags) + model_index:
                look_up_date = row['sales_date'] - pd.DateOffset(lag, 'D')
                if look_up_date in dates_in_buffer:
                    row[f'n_transactions_lag_{lag}'] = store_df_buffer[store_df_buffer['sales_date'] == pd.to_datetime(look_up_date)]['n_transactions'].values[0]
                elif look_up_date in dates_predicted.keys():
                    row[f'n_transactions_lag_{lag}'] = dates_predicted[look_up_date]
                else:
                    row[f'n_transactions_lag_{lag}'] = pd.NA 
            features_for_prediction = model.feature_name_
            features = row[features_for_prediction].to_frame().T
            #make sure that categorical features are in the same format as the training set
            for col in features.columns:
                if col in categorical_features:
                    features[col] = features[col].astype('category')
                else: 
                    features[col] = pd.to_numeric(features[col], errors='coerce')
            #predict the number of transactions

            prediction = model.predict(features)
            dates_predicted[row['sales_date']] = prediction[0]
            all_predictions_records.append({
                            'store_hashed': store,
                            'sales_date': row['sales_date'], 
                            'predicted_n_transactions': prediction[0],
                            'relative_horizon': i 
                        })
            
    predictions_df = pd.DataFrame(all_predictions_records)
    final_predictions_df = pd.merge(
            predictions_df,
            df_test[['store_hashed', 'sales_date', 'n_transactions']],
            on=['sales_date', 'store_hashed'],
            how='left' # Use left merge to keep all original test rows
        )
    return final_predictions_df