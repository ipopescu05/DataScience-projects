import pandas as pd
import numpy as np
import torch
from scipy import stats
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.utils import negative_sampling
from torch_geometric_temporal import recurrent
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, DynamicHeteroGraphTemporalSignal
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import HeteroGCLSTM
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GATConv, HeteroConv

import sys
sys.path.append('../utils')
from probabilistic_util_fuctions import generate_rfm_data2, generate_rfm_data



def data_preprocessing(df):

    df = df[df['CustomerID'].notna()].reset_index(drop=True)
    df = df[df.Quantity > 0]
    df = df[df.UnitPrice > 0]
    non_product_codes = ['POST', 'D', 'M', 'C2', 'BANK CHARGES', 'AMAZONFEE', 'DCGSSBOY', 
                         'DCGSSGIRL', 'DOT', 'PADS', 'TEST001','TEST002', 'ADJUST', 'ADJUST2','SP1002']

    df = df[~df['StockCode'].isin(non_product_codes)]

    #time features engeneering
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
    df['hour_minute'] = df['InvoiceDate'].dt.time

    df['day_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_cos'] = np.cos(2*np.pi*df['day_of_week']/7)

    morning_time = pd.to_datetime('12:00:00').time()
    afternoon_time = pd.to_datetime('18:00:00').time()
    df['time_of_day'] = df['hour_minute'].apply(lambda x: 'morning' if x < morning_time
                                                                        else 'afternoon' if (x < afternoon_time and x >= morning_time)
                                                                        else 'evening')
    
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    df = df[(np.abs(stats.zscore(df['TotalPrice'])) < 3)]
    df = df[(np.abs(stats.zscore(df['Quantity'])) < 3)]
    
    # create first order of the customer column
    df['FirstOrder'] = df.groupby('CustomerID')['InvoiceDate'].transform('min')


    #ebedding for the country and description columns
    country_encoder = LabelEncoder()
    df['CountryID'] = country_encoder.fit_transform(df['Country'])
    desc_encoder = LabelEncoder()
    df['DescriptionID'] = desc_encoder.fit_transform(df['Description'])
    #define embedding layers to be used in the model
    num_countries = df.CountryID.nunique()
    num_description = df.DescriptionID.nunique()
    country_embedding = nn.Embedding( num_embeddings=num_countries, embedding_dim = 4)
    description_embedding = nn.Embedding(num_embeddings=num_description, embedding_dim = 16)
    df = df[df.Month != '2011-12']

    return df, country_embedding, description_embedding

def get_orders_df(df): 
    rfm_data, orders_df= generate_rfm_data(df)
    orders_df1 = df.groupby('InvoiceNo').agg({
                                                'InvoiceDate': 'first',
                                                'Month': 'first',
                                                'hour_minute': 'first',
                                                'day_of_week': 'first',
                                                'day_sin': 'first',
                                                'day_cos': 'first',
                                                'StockCode': 'nunique',
                                                }).reset_index()
    orders_df = orders_df1.merge(orders_df, on='InvoiceNo', how='left')
    orders_df.rename(columns = {'StockCode':'unique_per_order'}, inplace = True)
    morning_time = pd.to_datetime('12:00:00').time()
    afternoon_time = pd.to_datetime('18:00:00').time()
    orders_df['time_of_day'] = orders_df['hour_minute'].apply(lambda x: 'morning' if x < morning_time
                                                                        else 'afternoon' if (x < afternoon_time and x >= morning_time)
                                                                        else 'evening')
    comon_type_od_day = orders_df.groupby('CustomerID')['time_of_day'].agg(lambda x:x.value_counts().idxmax()).reset_index(name = 'most_common_time_of_day')
    n_unique_items = df.groupby('CustomerID')['StockCode'].nunique().reset_index(name = 'n_unique_items')
    country_ID = df.groupby('CustomerID')['CountryID'].first().reset_index(name = 'CountryID')

    customer_df = orders_df.groupby('CustomerID')['unique_per_order'].mean().reset_index()
    customer_df = customer_df.merge(n_unique_items, on='CustomerID', how='left')
    customer_df = customer_df.merge(comon_type_od_day, on='CustomerID', how='left')
    customer_df = customer_df.merge(country_ID, on='CustomerID', how='left')
    customer_df = pd.get_dummies(customer_df, columns=['most_common_time_of_day'], prefix='in_').astype(int)
    customer_df = customer_df.merge(rfm_data, on='CustomerID', how='left')
    customer_df['avg_time_between_orders'] = customer_df['recency']/customer_df['frequency']
    customer_df['days_since_last_order'] = customer_df['T'] - customer_df['recency']
    customer_df['TotalPrice'] = customer_df['frequency'] * customer_df['monetary_value']

    return orders_df, customer_df

def create_dynamic_graph(df, start_t = 3, end_t = -3):
    """
    Constructs a dynamic heterogeneous graph dataset using a global node set.
    
    For each monthly snapshot, this function returns global feature matrices for 
    customers and products. Inactive nodes are filled with zeros. It also returns 
    a target vector for customers and a mask of active customer indices.
    """
    time_steps = sorted(df['Month'].unique())  # unique months

    edge_index_dicts = []
    edge_weights_dicts = []
    feature_dicts = []
    target_dicts = []

    # Create global mappings for customers and products

    customer_id_map = {customer_id: i for i, customer_id in enumerate(df['CustomerID'].unique())}
    product_id_map = {product_id: i for i, product_id in enumerate(df['StockCode'].unique())}
    df['global_cust_index'] = df['CustomerID'].map(customer_id_map)
    df['global_prod_index'] = df['StockCode'].map(product_id_map)


    total_customers = len(customer_id_map)
    total_products = len(product_id_map)

    for t, month in enumerate(time_steps[start_t:end_t]):
        snapshot = df[df['Month'] <= month].reset_index(drop=True)
        _, cust_features_df = get_orders_df(snapshot)
        customer_features = np.zeros((total_customers, 15))
        # Also, record which global customers are active in this snapshot
        active_customer_mask = np.zeros(total_customers, dtype=bool)
        cust_features_df['global_cust_index'] = cust_features_df['CustomerID'].map(customer_id_map)
    

        # Aggregate features for active customers from snapshot

        if t>0:
            one_month_ago = df[df['Month'] == time_steps[t - 1]]
            OMA_revenue = one_month_ago.groupby('global_cust_index')['TotalPrice'].sum().reset_index(name='lag1_revenue')
        else:
            OMA_revenue = pd.DataFrame(columns=['global_cust_index', 'lag1_revenue'])

        if t > 1:
            two_months_ago = df[df['Month'] == time_steps[t -2]]
            TMA_revenue = two_months_ago.groupby('global_cust_index')['TotalPrice'].sum().reset_index(name='lag2_revenue')
        else:
            TMA_revenue = pd.DataFrame(columns=['global_cust_index', 'lag2_revenue'])
        if t > 2:
            three_months_ago = df[df['Month'] == time_steps[t - 3]]
            ThMA_revenue = three_months_ago.groupby('global_cust_index')['TotalPrice'].sum().reset_index(name='lag3_revenue')
        else:
            ThMA_revenue = pd.DataFrame(columns=['global_cust_index', 'lag3_revenue'])
        cust_features_df = cust_features_df.merge(OMA_revenue, on='global_cust_index', how='left')
        cust_features_df = cust_features_df.merge(TMA_revenue, on='global_cust_index', how='left')
        cust_features_df = cust_features_df.merge(ThMA_revenue, on='global_cust_index', how='left')
        cust_features_df['lag1_revenue'] = cust_features_df['lag1_revenue'].fillna(0)
        cust_features_df['lag2_revenue'] = cust_features_df['lag2_revenue'].fillna(0)
        cust_features_df['lag3_revenue'] = cust_features_df['lag3_revenue'].fillna(0)
        
        cust_features_df = cust_features_df[['global_cust_index', 'lag1_revenue', 'lag2_revenue', 'lag3_revenue', 'unique_per_order', 'n_unique_items',
                                                'in__afternoon', 'in__evening', 'in__morning', 'frequency', 'recency',
                                                'T', 'monetary_value', 'avg_time_between_orders',
                                                'days_since_last_order', 'CountryID']]
        for _, row in cust_features_df.iterrows():
            global_id = int(row['global_cust_index'])
            # Fill in the features for this active customer
            customer_features[global_id, :] = row.drop('global_cust_index').values
            active_customer_mask[global_id] = True
        
        # Build a global product feature matrix of shape (total_products, 3)
        # Features: [Quantity, TotalPrice, DescriptionID]
        product_features = np.zeros((total_products, 5))
        active_product_mask = np.zeros(total_products, dtype=bool)
        
        prod_features_df = snapshot.groupby('global_prod_index').agg({
            "Quantity": "sum", 
            "TotalPrice": "sum",
            "UnitPrice": "mean",
            'global_cust_index': 'nunique',
            "DescriptionID": "first"
        }).reset_index()
        
        for _, row in prod_features_df.iterrows():
            global_id = int(row['global_prod_index'])
            product_features[global_id, :] = row.drop('global_prod_index').values
            active_product_mask[global_id] = True

        # Build edge indices using the global indices directly from the snapshot
        #snapshot_after_3_months = df_after_3_months[df_after_3_months['Month'] <= month].reset_index(drop=True)
        global_customer_indices = snapshot['global_cust_index'].values
        global_product_indices = snapshot['global_prod_index'].values
        edge_index = np.vstack((global_customer_indices, global_product_indices))
        
        # For edge weights, aggregate Quantity per edge (customer, product)
        aggregated_snapshot = snapshot.groupby(["global_cust_index", "global_prod_index"]).agg({
            "UnitPrice": "mean",
            "Quantity": "sum", 
            'day_sin': 'last',
            'day_cos': 'last'
        }).reset_index()
        edge_weight = aggregated_snapshot[['UnitPrice','Quantity', 'day_sin', 'day_cos']].values

        # Self-loops for customers: create self-edges for active customers
        active_cust_indices = np.where(active_customer_mask)[0]
        
        self_edge_index = np.vstack((active_cust_indices, active_cust_indices))
        self_edge_weight = np.ones(len(active_cust_indices))

        # Build the global target vector for customers (shape: total_customers)
        # We'll fill target values only for active customers; others remain 0.
        target = np.zeros(total_customers)
        # Compute future revenue for the next 3 months
        '''
        first_snapshot = df[df['Month'] == time_steps[t + 1]]
        second_snapshot = df[df['Month'] == time_steps[t + 2]]
        third_snapshot = df[df['Month'] == time_steps[t + 3]]
        first_revenue = first_snapshot.groupby('global_cust_index')['TotalPrice'].sum()
        second_revenue = second_snapshot.groupby('global_cust_index')['TotalPrice'].sum()
        third_revenue = third_snapshot.groupby('global_cust_index')['TotalPrice'].sum()
        
        
        for global_id in active_cust_indices:
            for future_revenue in [first_revenue, second_revenue, third_revenue]:
                if global_id in future_revenue.index:
                    target[global_id] += future_revenue[global_id]

        '''
        future_months = [time_steps[t + 1], time_steps[t + 2], time_steps[t + 3]]
        future_df = df[df['Month'].isin(future_months)]
        future_revenue = future_df.groupby('global_cust_index')['TotalPrice'].sum()
        target[future_revenue.index] = future_revenue.values

        # Append the active customer indices (mask) so we know which rows to evaluate
        active_customer_ids = active_cust_indices

        # Append data for this snapshot
        edge_index_dicts.append({
            ('customer', 'purchases', 'product'): edge_index,
            ('customer', 'self', 'customer'): self_edge_index
        })
        edge_weights_dicts.append({
            ('customer', 'purchases', 'product'): edge_weight,
            ('customer', 'self', 'customer'): self_edge_weight
        })
        feature_dicts.append({
            'customer': customer_features,
            'product': product_features
        })
        target_dicts.append({
            'customer': target, 
            'active_customer_ids': active_customer_ids
        })
        
    return edge_index_dicts, edge_weights_dicts, feature_dicts, target_dicts


def compute_normalization_param(train_dataset, node_type = 'customer', customer_numeric = 14, product_numeric = 4): 
    """
    Computes the normalization parameters (mean and std) for the customer features.
    """
    all_active_rows = []
    for snapshot in train_dataset:
        features = np.array(snapshot.x_dict['customer'])

        if node_type =='customer':
            active_ids = snapshot['active_customer_ids'].y
            cont_features = features[active_ids, :customer_numeric]
        elif node_type == 'product':
            cont_features = features[:, :product_numeric]
        else:
            raise ValueError("Invalid node type. Choose 'customer' or 'product'.")
        
        all_active_rows.append(cont_features)

    all_features = np.vstack(all_active_rows)
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    return scaler  

def compute_normalization_param_edge(train_dataset):
    """
    Computes the normalization parameters (mean and std) for the edge features.
    """
    all_active_rows = []
    for snapshot in train_dataset:
        features = np.array(snapshot[('customer', 'purchases', 'product')].edge_attr)

        all_active_rows.append(features)

    all_features = np.vstack(all_active_rows)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler


def normalize_snapshot(snapshot, customer_scaler, product_scaler, edge_scaler, customer_numeric = 14, product_numeric = 4): 

    customer_features = snapshot.x_dict['customer']
    active_customer_ids = snapshot['active_customer_ids'].y
    product_features = snapshot.x_dict['product']

    # Normalize customer features
    snapshot.x_dict['customer'][active_customer_ids, :customer_numeric] = torch.tensor(customer_scaler.transform(customer_features[active_customer_ids, :customer_numeric]), dtype=torch.float)

    # Normalize product features
    snapshot.x_dict['product'][:, :product_numeric] = torch.tensor(product_scaler.transform(np.array(product_features[:, :product_numeric])), dtype=torch.float)
    # Normalize edge features
    edge_weights = snapshot[('customer', 'purchases', 'product')].edge_attr
    snapshot[('customer', 'purchases', 'product')].edge_attr = torch.tensor(edge_scaler.transform(np.array(edge_weights)), dtype=torch.float)
    return snapshot



def normalize_dataset(dataset, scaler_customer, scaler_product, scaler_edge):
    list_of_normalized_snapshots = []
    for snapshot in dataset:
        normalized_snapshot = normalize_snapshot(snapshot, scaler_customer, scaler_product, scaler_edge)
        list_of_normalized_snapshots.append(normalized_snapshot)
    return list_of_normalized_snapshots