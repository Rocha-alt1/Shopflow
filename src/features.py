import pandas as pd
import numpy as np

def build_features(df, prediction_date):

    date = pd.to_datetime(prediction_date) # just making sure we dont have a problem with the input 
    
    #making sure the data is clean and splited into our timestamp
    clean_df = df[(df['Quantity'] > 0) & (df['CustomerID'].notnull())]
    feature_df = clean_df[clean_df['InvoiceDate'] < date].copy()

    #creates the new variables
    feature_recency = feature_df.groupby('CustomerID')['InvoiceDate'].max()
    feature_recency = (date - feature_recency).dt.days.rename('recency')
                       
    feature_orders = feature_df.groupby('CustomerID')['InvoiceNo'].nunique().rename('total_orders')
    feature_spent = feature_df.groupby('CustomerID')['revenue'].sum().rename('total_spent')
    avg_order = (round(feature_spent / feature_orders, 2)).rename('avg_order')

    feature_df['months'] = feature_df['InvoiceDate'].dt.to_period('M')
    purchased_months = feature_df.groupby('CustomerID')['months'].nunique().rename('purchased_months')

    final_df = pd.concat([feature_recency, feature_orders, feature_spent, avg_order, purchased_months], axis=1)

    return final_df




def build_target(df, prediction_date, days=30):

    date = pd.to_datetime(prediction_date) 
    time = date + pd.Timedelta(days=days) 

    clean_df = df[(df['Quantity'] > 0) & (df['CustomerID'].notnull())]

    if clean_df['InvoiceDate'].max() < time:
        raise ValueError("The full target window is not available.")

    # Purchases made during the target window
    target_df = clean_df[(clean_df['InvoiceDate'] >= date) & (clean_df['InvoiceDate'] < time)]

    # One positive label per customer who purchased
    target1 = (target_df.groupby('CustomerID').size().rename('target'))

    target1[:] = 1

    # Customers with features before the prediction date
    feature_customers = clean_df[clean_df['InvoiceDate'] < date]['CustomerID'].unique()

    # Customers who did not purchase during the window get 0
    targets = pd.Series(
        0,
        index=pd.Index(feature_customers, name='CustomerID'),
        name='target'
    )

    # Mark customers who purchased as 1
    targets.loc[targets.index.intersection(target1.index)] = 1

    return targets