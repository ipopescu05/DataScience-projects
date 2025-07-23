from lifetimes import ParetoNBDFitter
from lifetimes import BetaGeoFitter
import lifetimes
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.utils import summary_data_from_transaction_data
from pymc_marketing import clv
import pandas as pd

def generate_rfm_data (df, last_transaction_date = pd.to_datetime('2011-12-09')):
    agg_function = {
    'CustomerID': 'first',
    'Amount': 'sum',
    'Date': 'min',
    'Country': 'first'}

    orders_df = df.groupby('InvoiceNo').agg(agg_function).reset_index()
    orders_df.drop(orders_df.index[orders_df.Amount <=0], axis = 0, inplace = True)
    rfm = summary_data_from_transaction_data(transactions=orders_df,
                                         customer_id_col='CustomerID',
                                         datetime_col='Date',
                                         monetary_value_col = 'Amount',
                                         observation_period_end=last_transaction_date,
                                         freq='D',
                                        include_first_transaction = True)
    return rfm, orders_df

def generate_rfm_data2 (df, last_transaction_date = pd.to_datetime('2011-12-09')):
    agg_function = {
    'global_cust_index': 'first',
    'TotalPrice': 'sum',
    'InvoiceDate': 'min',
    'Country': 'first'}

    orders_df = df.groupby('InvoiceNo').agg(agg_function).reset_index()
    orders_df.drop(orders_df.index[orders_df.TotalPrice <=0], axis = 0, inplace = True)
    rfm = summary_data_from_transaction_data(transactions=orders_df,
                                         customer_id_col='global_cust_index',
                                         datetime_col='InvoiceDate',
                                         monetary_value_col = 'TotalPrice',
                                         observation_period_end=last_transaction_date,
                                         freq='D',
                                        include_first_transaction = True)
    return rfm, orders_df

def get_orders_df (df):
    agg_function = {
    'CustomerID': 'first',
    'Amount': 'sum',
    'Date': 'min',
    'Country': 'first'}

    orders_df = df.groupby('InvoiceNo').agg(agg_function).reset_index()
    orders_df.drop(orders_df.index[orders_df.Amount <=0], axis = 0, inplace = True)
    return orders_df

def generate_calibration_and_holdout (df, last_transaction_date = '2011-12-09', offset = 30 ):
    orders_df = get_orders_df(df)
    rfm = calibration_and_holdout_data(transactions=orders_df,
                                         customer_id_col='CustomerID',
                                         datetime_col='Date',
                                         monetary_value_col = 'Amount',
                                         freq = 'D',
                                         calibration_period_end=pd.to_datetime(last_transaction_date) - pd.DateOffset(days=offset),
                                         observation_period_end=pd.to_datetime(last_transaction_date)
                                        ).reset_index()
    data = (
    rfm[[ "CustomerID", 'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal']]
    .rename(columns={'CustomerID':'customer_id',
                    'frequency_cal':'frequency',
                      'recency_cal':'recency',
                    'monetary_value_cal':'monetary_value',
                     'T_cal': 'T'
                    })

    )
    data_holdout = (
    rfm[[ "CustomerID", 'frequency_holdout', 'monetary_value_holdout']]
    .rename(columns={'CustomerID':'customer_id',
                     'frequency_holdout':'frequency',
                     'monetary_value_holdout':'monetary_value'
                     })
    )
    return data, data_holdout

def generate_calibration_and_holdout_2 (df, last_transaction_date = '2011-12-09', offset = 30 ):
   orders_df = get_orders_df(df)
   rfm = clv.utils.rfm_train_test_split(transactions=orders_df,
                                         customer_id_col='CustomerID',
                                         datetime_col='Date',
                                         monetary_value_col = 'Amount',
                                         time_unit = 'D',
                                         train_period_end=pd.to_datetime(last_transaction_date) - pd.DateOffset(days=offset),
                                         test_period_end=pd.to_datetime(last_transaction_date)
                                        ).reset_index()
   data = rfm[[ "customer_id", 'frequency', 'recency', 'T', 'monetary_value']]
   data_holdout = (
    rfm[[ "customer_id", 'test_frequency', 'test_monetary_value']]
    .rename(columns={
                     'test_frequency':'frequency',
                     'test_monetary_value':'monetary_value'
                     })
    )
   return data, data_holdout

def transaction_churn_model_lt(data, prob_model = 'ParetoNBD', library  = 'pymc'):
    if library != 'pymc':
        if prob_model == 'ParetoNBD':
            model = ParetoNBDFitter()
        else:
          model = lifetimes.BetaGeoFitter(penalizer_coef=0.0)

        model.fit(data['frequency'], data['recency'], data['T'])
        return model
    if prob_model == 'ParetoNBD':
        model = clv.ParetoNBDModel(data = data)
    else:
        model =clv.BetaGeoModel(data = data)
    model.fit()
    return model

def predict_trasactions_churn(data, model, future_t = 52,  library = 'pymc'):
  if library != 'pymc':
    number_of_transactions = model.conditional_expected_number_of_purchases_up_to_time(t = future_t,
                        frequency = data['frequency'],
                        recency = data['recency'],
                         T = data['T'])
    alive_prob = model.conditional_probability_alive(
                        frequency = data['frequency'],
                        recency = data['recency'],
                         T = data['T'])

  else:
    number_of_transactions = model.expected_purchases(future_t = future_t, data = data).mean(axis=(0, 1))
    alive_prob = model.expected_probability_alive(data = data).mean(axis = (0, 1))
  return number_of_transactions, alive_prob


def gg_model(rfm):
  nonzero_data = rfm.query('frequency > 0')
  print('monetary value - frequency correlation: ', nonzero_data[["monetary_value", "frequency"]].corr())

  gg = clv.GammaGammaModel(data = nonzero_data)
  gg.build_model()
  gg.fit()
  return gg

def predict_clv(rfm, model, future_t):
  expected_spend = model.expected_customer_spend(data = rfm)
  expected_monetary_value = expected_spend.mean(axis = (0, 1)).to_dataframe(name = 'expected_spend').reset_index()
  return expected_monetary_value