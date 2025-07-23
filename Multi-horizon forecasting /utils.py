
import pandas as pd 
import numpy as np
from global_model_util import recursive_forecast



# FUNCTIONS USED IN EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING
def stores_with_operational_breaks(df, break_days=7): 
    """
    This function takes a DataFrame and returns a list of stores that have operational breaks.
    An operational break is defined as a minimum of prespecified number of consecutive days without transactions.

    Input:
    - df: DataFrame containing sales data with 'store_hashed' and 'sales_date' columns.
    - break_days: Number of consecutive days without transactions to define an operational break.
    Output:
    - store_ids_with_breaks: List of unique store IDs with operational breaks.
    - store_ids_with_multiple_breaks: List of unique store IDs with more than one operational break.
    """
    
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df = df.sort_values(by=['store_hashed', 'sales_date'])
    
    # Group by store and calculate the difference in days between consecutive sales dates
    df['date_diff'] = df.groupby('store_hashed')['sales_date'].diff().dt.days
    
    # Identify breaks of more than the specified number of days
    breaks = df[df['date_diff'] > break_days]
    
    # Get the unique store hash with operational breaks
    store_ids_with_breaks = breaks['store_hashed'].unique()

    #stores with more than 1 break
    store_ids_with_multiple_breaks = breaks['store_hashed'].value_counts()[breaks['store_hashed'].value_counts() > 1].index.tolist()
    
    return store_ids_with_breaks, store_ids_with_multiple_breaks




def preprocess_data(df):
    """
    Preprocess the data by removing stores that are not operational anymore, stores with multiple operational breaks longer than 14 days, and observations with no running hours data.
    Also, engineer the running hours features and additional time features. 
    INPUT:
     - df: pd.DataFrame with original data
    OUTPUT:
     - df: pd.DataFrame with preprocessed data
    """
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df = df.sort_values(['store_hashed', 'sales_date'])

    # remove stores that are not operational anymore
    last_operating_day = df.groupby('store_hashed')['sales_date'].max().reset_index()
    non_operating_stores_ids = last_operating_day[last_operating_day['sales_date'] < pd.to_datetime('2022-12-22')]['store_hashed'].unique()
    df = df[~df['store_hashed'].isin(non_operating_stores_ids)]
    
    # remove stores with multiple operational breaks longer than 14 days
    _, multiple_breaks_14 = stores_with_operational_breaks(df, break_days=14)
    df = df[~df['store_hashed'].isin(multiple_breaks_14)]
    # remove observations with no running hours data. They are unusual and I decided to remove them
    df = df.dropna(subset=['datetime_store_open', 'datetime_store_closed'])
    
    # engeneer the running hours features
    df['running_hours'] = df['running_hours'] = (df['datetime_store_closed'] - df['datetime_store_open']).dt.total_seconds() / 3600
    df['openning_hour'] = df['datetime_store_open'].dt.hour
    df['closing_hour'] = df['datetime_store_closed'].dt.hour
   
   # engeneer additional time features
    df['day_of_week'] = df['sales_date'].dt.dayofweek
    df['day_of_month'] = df['sales_date'].dt.day
    df['day_of_year'] = df['sales_date'].dt.dayofyear
    df['week_of_year'] = df['sales_date'].dt.isocalendar().week
    df['month'] = df['sales_date'].dt.month
    df['is_weekend'] = (df['day_of_week'] > 5).astype(int)

    
    df['is_national_holiday'] = df[['holiday_saint_nicholas',
       'holiday_first_christmas', 'holiday_liberation_day',
       'holiday_good_friday', 'holiday_new_years_day',  'holiday_kings_day_netherlands',
       'holiday_kings_day_belgium', ]].sum(axis=1)
    df['Covid19'] = ((df['sales_date'] >= pd.to_datetime('2020-03-01')) & (df['sales_date'] <= pd.to_datetime('2021-07-31'))).astype(int)

    return df


def split_train_test(df, number_of_days = 50):
    '''
    Split the data into train and test sets. The nimum number of days in the test set should be 50 days. 
    The function accounts for the fact that the stores have different number of days and different last sales dates.
    Input: 
    - df: pd.DataFrame with preprocessed data
    - number_of_days: the number of days considered for the test set
    Output:
    - df_train: pd.DataFrame with train data
    - df_test: pd.DataFrame with test data
    ''' 
    if 'is_test' in df.columns:
        df = df.drop(columns=['is_test'])
    df = df.sort_values(['store_hashed', 'sales_date'])
    df['is_test'] = False
    for store in df['store_hashed'].unique():
        store_df = df[df['store_hashed'] == store]
        if len(store_df) > number_of_days:
            df.loc[store_df.index[-number_of_days:], 'is_test'] = True
    return df[df['is_test'] == False], df[df['is_test'] == True]

def weighted_average(last_weeks_sales, last_year_sales, holiday_weight = 0.3):
    """
    This function is used in the context of the naive weighted forecast.
    Perform the weighted average of the last year sales and the last weeks sales.
    The weight is decaying with the distance to the current week.
    """
    m_weeks = last_weeks_sales.size
    weights = np.arange(1, m_weeks+1)*1/np.arange(1, m_weeks+1).sum()

    # if the weighted average consideres a national holiday, its weight is 0.3
    # the remaining weight is distributed to the last weeks sales in a decaying way
    if last_year_sales.size > 0:
        return holiday_weight*last_year_sales + (1-holiday_weight)*weights.dot(last_weeks_sales)
    return weights.dot(last_weeks_sales)

def naive_weighted_forecast(df_train, df_test, m_weeks = 8, pred_horizon = 50):
    '''
    Naive weighted forecast.
    The next step forecast is a weighted average of the same weekday of the last m_weeks 
    and the same day of the last year if it was a free national day (exemple last day of the year or Christmas day). 
    Input: 
    - df_train: pd.DataFrame with train data
    - df_test: pd.DataFrame with test data
    - m_weeks: the number of weeks considered for the forecast
    - pred_horizon: the number of days to forecast
    Output:
    - df_test_set: pd.DataFrame with test data and predictions
    '''
    #get the observations corresponding to the prediction horizon in the test set
    df_test_set = df_test.copy()  
    if 'relative_horizon' in df_test_set.columns:
        df_test_set = df_test_set.drop(columns=['relative_horizon'])                                       
    df_test_set = df_test_set.rename(columns = {'n_transactions': 'actual_n_transactions'})
    df_test_set['n_transactions'] = None

    # the algorithm is applied to each store separately
    for store in df_train['store_hashed'].unique():

        store_df_train = df_train[df_train['store_hashed'] == store].copy()
        store_df_test = df_test_set[df_test_set['store_hashed'] == store][:pred_horizon].copy()

        for i, row in store_df_test.iterrows():
            # prediction if national holiday
            last_year_sales = np.array([]) 
            if row['is_national_holiday'] > 0: 
                last_year_sales = store_df_train[store_df_train['sales_date'] == row['sales_date'] - pd.DateOffset(years=1)]['n_transactions'].values
            #get previous weeks sales
            current_date = row['sales_date']
            current_day = row['day_of_week']
            last_weeks_sales = store_df_train[(store_df_train['day_of_week'] == current_day)&
                                      (store_df_train['sales_date'] < current_date)]['n_transactions'].values
            if last_weeks_sales.size < m_weeks:
                last_weeks_sales = last_weeks_sales[-last_weeks_sales.size:]
            else:
                last_weeks_sales = last_weeks_sales[-m_weeks:]
            row['n_transactions'] = weighted_average(last_weeks_sales, last_year_sales)
            df_test_set.loc[i, 'n_transactions'] = row['n_transactions']
            store_df_train = pd.concat([store_df_train, row.drop('actual_n_transactions').to_frame().T], ignore_index=True)
    
    df_test_set['relative_horizon'] = df_test_set.groupby('store_hashed').cumcount() + 1

    return df_test_set[['store_hashed', 'sales_date', 'n_transactions', 'actual_n_transactions', 'relative_horizon']][df_test_set['relative_horizon']<=pred_horizon].copy()

def metrics_accross_horizon(df_test_with_predictions, pred_horizon = 50, naive = False):
    '''
    This function calculates the average RMSE, MAE, SMAPE and MAPE for each forecasting horizon, accross all the stores.
    Input:
    - df_test_with_predictions: pd.DataFrame with test data and predictions
    - pred_horizon: the number of forecasting days
    - naive: if True, the function calculates the metrics for the naive forecast. If False, it calculates the metrics for the recursive forecast.
    Output:
    - rmse_values: np.array with the RMSE values for each forecasting horizon
    - mae_values: np.array with the MAE values for each forecasting horizon
    - smape_values: np.array with the SMAPE values for each forecasting horizon
    - mape_values: np.array with the MAPE values for each forecasting horizon
    '''
    rmse_values = np.zeros(pred_horizon)
    mae_values = np.zeros(pred_horizon)
    smape_values = np.zeros(pred_horizon)
    mape_values = np.zeros(pred_horizon)
    if not naive:
        for h in range(pred_horizon):
            df_h = df_test_with_predictions[df_test_with_predictions['relative_horizon'] == h+1].copy()
            rmse_values[h] = np.sqrt(np.mean((df_h['predicted_n_transactions'] - df_h['n_transactions'])**2))
            mae_values[h] = np.mean(np.abs(df_h['predicted_n_transactions'] - df_h['n_transactions']))
            mape_values[h] = np.mean(np.abs((df_h['predicted_n_transactions'] - df_h['n_transactions'])/df_h['n_transactions'])*100)
            smape_values[h] = np.mean(np.abs(df_h['predicted_n_transactions'] - df_h['n_transactions'])/((np.abs(df_h['predicted_n_transactions'])+np.abs(df_h['n_transactions']))/2)*100)
        return rmse_values, mae_values,mape_values,  smape_values

    for h in range(pred_horizon):
        df_h = df_test_with_predictions[df_test_with_predictions['relative_horizon'] == h+1].copy()
        rmse_values[h] = np.sqrt(np.mean((df_h['n_transactions'] - df_h['actual_n_transactions'])**2))
        mae_values[h] = np.mean(np.abs(df_h['n_transactions'] - df_h['actual_n_transactions']))
        mape_values[h] = np.mean(np.abs((df_h['n_transactions'] - df_h['actual_n_transactions'])/df_h['actual_n_transactions'])*100)
        smape_values[h] = np.mean(np.abs(df_h['n_transactions'] - df_h['actual_n_transactions'])/((np.abs(df_h['actual_n_transactions'])+np.abs(df_h['n_transactions']))/2)*100)
    return rmse_values, mae_values, mape_values, smape_values

def add_lags(df, lags): 
    """
    Add lags to the training dataframe. This function accounts for the fact there are missing dates in the data.
    Input: 
    - df: pd.DataFrame with the training data
    - lags: list of lags to add
    Output:
    - df_to_change: pd.DataFrame with the training data and the lags added
    """
    df_to_change = df.copy()
    df_indexed = df_to_change.set_index(['store_hashed','sales_date'])['n_transactions']

    for lag in lags: 
        feature_name = f'n_transactions_lag_{lag}'

        # get the dates corresponding to the lags
        lookup_dates = df_to_change['sales_date'] - pd.DateOffset(lag, 'D')

        # find the index of the dates in the original dataframe
        lookup_multi_index = pd.MultiIndex.from_arrays([df_to_change['store_hashed'], lookup_dates])

        # reindex the original dataframe to get the values for the lags
        df_to_change[feature_name] = df_indexed.reindex(lookup_multi_index).values

    return df_to_change

def rolling_window_evaluation(df_test, 
                            df_train = None,
                            buffer = None,
                            model = 'naive',
                            m_weeks = 26, 
                            pred_horizon = 50,
                            window_size = 5 , 
                            lags = [1, 2, 3, 4, 5, 6, 7,8, 14, 21, 28, 35, 42, 49] ):
    """
    Thi function performs a rolling window evaluation of the model. For each considered point in time, the model is used to predict the next pred_horizon days.
    Then the global average RMSE, MAE, SMAPE and MAPE are calculated for each window. The global average is considered to be the average accross all the stores and all time horizon. 

    INPUT: 
    - df_train: the training set, set to None for the global ML models
    - buffer: the buffer set, set to None for the naive forecast
    - model: the model to use for the forecast. 'naive' for the naive forecast, 'recursive' for the recursive forecast
    - df_test: the test set
    - m_weeks: the number of weeks to use for the forecast
    - pred_horizon: the number of days to forecast
    - window_size: the size of the rolling window
    OUTPUT:
    - general_rmse_values: the global average RMSE values for each window
    - general_mae_values: the MAE values for each window
    - general_smape_values: the SMAPE values for each window
    - general_mape_values: the MAPE values for each window
    """
    #initialize the vectors to store the results
    general_rmse_values = list()
    general_mae_values = list()
    general_smape_values = list()
    general_mape_values = list()


    if df_train is not None:
        local_df_train = df_train.copy()
    elif buffer is not None: 
        local_buffer = buffer.copy()
        
    df_test_set = df_test.copy()

    #add the relative horizon to the test set. It will be used to easily roll the window forward
    df_test_set['relative_horizon'] = df_test_set.groupby('store_hashed', observed = True).cumcount() + 1
    forecast_window = df_test_set['relative_horizon'].max() - pred_horizon
    iter = 0

    local_df_test =  df_test_set[df_test_set['relative_horizon'] <= pred_horizon + window_size*iter].copy()

    #while there are still predictions to be made
    if model =='naive':
        while forecast_window > 0:    
            local_df_train = local_df_train.sort_values(['store_hashed', 'sales_date'])

            #get predictions for the current iteration
            test_with_predictions = naive_weighted_forecast(local_df_train, local_df_test, m_weeks = m_weeks, pred_horizon = pred_horizon)

            #store the predictions in the test set
            rmse_values, mae_values, mape_values, smape_values = metrics_accross_horizon(test_with_predictions, pred_horizon=pred_horizon, naive = True)
            general_rmse_values.append(rmse_values.mean())
            general_mae_values.append(mae_values.mean())
            general_smape_values.append(smape_values.mean())
            general_mape_values.append(mape_values.mean())

            #update the train and the test set (roll the window forward): take the first window_size from the test set and add them to the train set
            local_df_train = pd.concat([local_df_train, local_df_test[local_df_test['relative_horizon'] <= window_size*(iter+1)]], ignore_index=True)
            local_df_test = df_test_set[(df_test_set['relative_horizon'] > window_size*(iter+1))&(df_test_set['relative_horizon'] <= pred_horizon + window_size*(iter+1))].copy()
            

            forecast_window -= window_size
            iter += 1
            
        return general_rmse_values, general_mae_values, general_smape_values, general_mape_values
    else:
        while forecast_window > 0:
            local_buffer = local_buffer.sort_values(['store_hashed', 'sales_date'])
            #get predictions for the current iteration
            test_with_predictions = recursive_forecast(local_buffer, local_df_test, model = model,lags = lags )
            #store the predictions in the test set
            rmse_values, mae_values, mape_values, smape_values = metrics_accross_horizon(test_with_predictions, pred_horizon=pred_horizon, naive = False)
            general_rmse_values.append(rmse_values.mean())
            general_mae_values.append(mae_values.mean())
            general_smape_values.append(smape_values.mean())
            general_mape_values.append(mape_values.mean())
            #update the train and the test set (roll the window forward): take the first window_size from the test set and add them to the train set
            local_buffer  = pd.concat([local_buffer, local_df_test[local_df_test['relative_horizon'] <= window_size*(iter+1)]], ignore_index=True) 
            local_df_test = df_test_set[(df_test_set['relative_horizon'] > window_size*(iter+1))&(df_test_set['relative_horizon'] <= pred_horizon + window_size*(iter+1))].copy()
            forecast_window -= window_size
            iter += 1
            print(f"iteration {iter} done, {forecast_window/window_size} windows left")
            print(f"RMSE: {rmse_values.mean()}, MAE: {mae_values.mean()}, SMAPE: {smape_values.mean()}, MAPE: {mape_values.mean()}")

        return general_rmse_values, general_mae_values, general_smape_values, general_mape_values