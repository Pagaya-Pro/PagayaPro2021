import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy_financial as npf
import os
import warnings
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')
from collections import Counter
from tqdm import tqdm
import datetime as dt
import xgboost as xgb
import prod_infra as infra
import subprocess
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
plt.style.use('seaborn')



RANDOM_SEED = 42


def load_data():
    '''
    load parquets 
    '''
    PROJECT_PATH = '/mnt/s3/pagaya-pro-bucket/projects/re_recommendation_system/*.parquet'
    files = glob.glob('/mnt/s3/pagaya-pro-bucket/projects/re_recommendation_system/*.parquet') 
    data = pd.read_parquet(files)
    return data 

def preprocess_data(data):
    '''
        preprocess, filter by features, and prepare features
    '''
    data_sfr = infra.preprocess(data)
    data_sfr = infra.filter_by_features(data_sfr)
    data_sfr_pre,target = infra.prepare_data(data_sfr)
    return data_sfr_pre, target

def fixed_xgb_actual_pred_error_df(comp_price_distance, idx_comp, X, comps, target_tol = 'price', X_sqft = None, comps_sqft = None):
    '''
    crate actual/predicted df (including error and error percent) - given asset and comps predict the actual as the comps mean
    comp_price_distance : change in price according to XGB
    idx_comp: dict - key is the asset id, value is list of the comps id
    X: df/series with assets prices
    comps: df/series with the comps prices
    target_tol: the column name in the comps/X of the price(used for price/price sqft)
    
    in case the target_tol == price_sqft we need to adjust the price :
    X_sqft: df/series with assets  sqft 
    comps_sqft: df/series with the comps sqft
    '''
    actucal_price_sqf, pred_price_sqf = [], []
    if target_tol == 'price_sqf':
        return actual_pred_df_sqft(idx_comp, X, comps, X_sqft, comps_sqft )
    for key in idx_comp:
        actucal_price_sqf.append(X.loc[key]['price'])
        pred_price_sqf.append((comps.loc[idx_comp[key]]['price'] + comp_price_distance[key]).mean()) 
        
    df = pd.DataFrame({'actual':actucal_price_sqf,'pred':pred_price_sqf})
    df['error'] = df.pred - df.actual 
    df['error_pcnt'] = df['error'] / df['actual'] * 100
    
    return df


def fixed_weighted_xgb_actual_pred_error_df(comp_price_distance, idx_comp, X, comps, target_tol = 'price', X_sqft = None, comps_sqft = None):
    '''
    crate actual/predicted df (including error and error percent) - given asset and comps predict the actual as the comps mean
    comp_price_distance : change in price according to XGB
    idx_comp: dict - key is the asset id, value is list of the comps id
    X: df/series with assets prices
    comps: df/series with the comps prices
    target_tol: the column name in the comps/X of the price(used for price/price sqft)
    
    in case the target_tol == price_sqft we need to adjust the price :
    X_sqft: df/series with assets  sqft 
    comps_sqft: df/series with the comps sqft
    '''
    actucal_price_sqf, pred_price_sqf = [], []
    if target_tol == 'price_sqf':
        return actual_pred_df_sqft(idx_comp, X, comps, X_sqft, comps_sqft )
    for key in idx_comp:
        actucal_price_sqf.append(X.loc[key]['price'])
        fixed = comps.loc[idx_comp[key]]['price'] + comp_price_distance[key]
        pred_price_sqf.append(np.average(fixed, weights=fixed/fixed.sum())) 
        
    df = pd.DataFrame({'actual':actucal_price_sqf,'pred':pred_price_sqf})
    df['error'] = df.pred - df.actual 
    df['error_pcnt'] = df['error'] / df['actual'] * 100
    
    return df
    
    
def actual_pred_error_df(idx_comp, X, comps, target_tol = 'price', X_sqft = None, comps_sqft = None, comp_price_distance=None):
    '''
    crate actual/predicted df (including error and error percent) - given asset and comps predict the actual as the comps mean
    idx_comp: dict - key is the asset id, value is list of the comps id
    X: df/series with assets prices
    comps: df/series with the comps prices
    target_tol: the column name in the comps/X of the price(used for price/price sqft)
    
    in case the target_tol == price_sqft we need to adjust the price :
    X_sqft: df/series with assets  sqft 
    comps_sqft: df/series with the comps sqft
    '''
    actucal_price_sqf, pred_price_sqf = [], []
    if target_tol == 'price_sqf':
        return actual_pred_df_sqft(idx_comp, X, comps, X_sqft, comps_sqft )
    for key in idx_comp:
        actucal_price_sqf.append(X.loc[key]['price'])
        pred_price_sqf.append(comps.loc[idx_comp[key]]['price'].mean()) 
        
    df = pd.DataFrame({'actual':actucal_price_sqf,'pred':pred_price_sqf})
    df['error'] = df.pred - df.actual 
    df['error_pcnt'] = df['error'] / df['actual'] * 100
    
    return df


def actual_pred_df_sqft(idx_comp, X, comps, X_sqft, comps_sqft):
    '''
    auxiliary function for add_error_df
    '''
    actucal_price_sqf, pred_price_sqf = [], []
    for key in idx_comp:
        actucal_price_sqf.append(X.loc[key].price_sqf*X_sqft.loc[key])
        price = (comps.loc[idx_comp[key]].price_sqf* comps_sqft.loc[idx_comp[key]]).mean()
        pred_price_sqf.append(price)
        
    df = pd.DataFrame({'actual':actucal_price_sqf,'pred':pred_price_sqf})
    df['error'] = df.pred - df.actual 
    df['error_pcnt'] = df['error'] / df['actual'] * 100
    
    return df


def plot_actual_pred(pred_error_df, title="Predicted Price vs. Actual Price"):
    '''
    prints the Mdape + plots the actual vs predicted
    pred_error_df: output df from actual_pred_error_df function
    '''
    plt.style.use('seaborn')
    print(f"We received an MdAPE of {pred_error_df['error_pcnt'].abs().median():.3f}% when predicting {len(pred_error_df)} home prices\n")
    plt.figure(figsize=(15,9))
    pred_error_df = pred_error_df[pred_error_df.actual<1000000]
    sns.regplot(data = pred_error_df, x='actual' , y='pred', label='predictions', line_kws={'label': 'regplot'},)
    plt.title(title, fontweight='bold', fontsize=16)
    line = [pred_error_df.min().min(), pred_error_df.max().max()]
    sns.lineplot(x=line, y=line, label='x=y')
    plt.title(title, size=16, weight='bold')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price");
    plt.ylim(0)
    plt.xlim(0);
    #plt.gca().invert_yaxis()
    
    
def train_test_split(data, target, city, date):
    '''
        train test split  - OOT
        data: DataFrame
        target: target
        date: string map to use
        city: string city to map 
        filter to city, split the data into train and test starting from date
        train is all the samples starting from date to date+6 monthes
        test is one month after the train end
    '''
    date = dt.datetime.strptime(date,'%Y-%m-%d')
    city_df = data[data.city_correct == city]
    city_df = city_df[city_df.date >= date]
    city_df = city_df.sort_values(by='date')
    # add 6 months 
    end_train_date = date + dt.timedelta(days=6*30)
    end_test_date = end_train_date + dt.timedelta(days=2*30)
    city_df_train = city_df[city_df.date <= end_train_date]
    city_df_test = city_df[(city_df.date > end_train_date) & (city_df.date < end_test_date)]
    return city_df_train, target.loc[city_df_train.index], city_df_test, target.loc[city_df_test.index]


def normalize_features(df, STANDARD_SCALING_COLS = ['bedrooms', 'gla_sqft', 'year_built', 'full_baths', 'property_longitude',
                             'property_latitude', 'bath_to_room', 'sqft_to_room', 'month']):
    '''
        STANDARD_SCALING_COLS: list of cols to norm
        StandardScaler normalization to STANDARD_SCALING_COLS,
        cols that are in df but not in STANDARD_SCALING_COLS are part of the returnd scaled_df
    '''    

    STANDARD_SCALING_COLS = STANDARD_SCALING_COLS
    
    other = df.columns[~df.columns.isin(STANDARD_SCALING_COLS)]

    scaled_df = df[[*STANDARD_SCALING_COLS]].copy()

    standard_scaler = StandardScaler()
    scaled_df[STANDARD_SCALING_COLS] = standard_scaler.fit_transform(scaled_df[STANDARD_SCALING_COLS]) #* weights
    
    scaled_df[other] = df[other]

    return scaled_df



def preprocess_data_kmeans(data_sfr_pre, target, city='Houston', date = '2020-01-01'):
    '''
        preprocess, filter by features, normalize and prepare features
    '''
    houston_df = data_sfr_pre[data_sfr_pre.city_correct == city]
    houston_df = houston_df[houston_df.date > date]
    target_houston = target.loc[houston_df.index]
    houston_df_norm = normalize_features(houston_df)
    k_means_data = houston_df_norm.drop(['date','year','city_correct','property_latitude', 'property_longitude','zip'], axis=1)
    to_float = ['gla_sqft', 'bedrooms', 'full_baths', 'month', 'year_built', 'has_spa','has_fireplace']
    k_means_data[to_float]= k_means_data[to_float].astype('float64')
    return k_means_data, target_houston