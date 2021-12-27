import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from geopy.distance import distance as geodist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings('ignore')

# def normalize_features(df, cols):
#     scaler = StandardScaler()
#     scaled_df = scaler.fit_transform(df[cols])
#     return scaled_df

def preprocess(df):
    
    # Drop columns with more than 90 percent nan values 
    isna = df.isna().sum()
    col_to_drop = isna[isna > df.shape[0]*0.9]
    df = df.drop(col_to_drop.index, axis=1)
    
    # Create year column
    df['year'] = df.date.dt.year
    
    # Filter SFR houses
    prop_types = ['Single Family Detached',
                             'Single Residential',
                             'Detached Single',
                             'Single Family',
                             'Single-Family',
                             'Res - Sngl Fam']
    data_sfr = df.loc[df['mls_property_type'].isin(prop_types)]
    data_sfr = data_sfr[~data_sfr.mls_property_sub_type.str.contains('Duplex')]
    data_sfr = data_sfr[~(data_sfr['sale_type'] == 'Lease')]
    
    # Fill correct cities using zip codes
    df_from_web = pd.read_csv('../../../../Downloads/uszips.csv')
    df_from_web['zip'] = df_from_web['zip'].astype('str')
    data_sfr['city_correct'] = data_sfr['zip'].map(df_from_web.set_index('zip')['city'])
    
    # Filter top 15 cities by listings count
    top_cities = data_sfr['city_correct'].value_counts().head(15).index
    data_sfr = data_sfr[data_sfr.city_correct.isin(top_cities)]
    
    # Drop latitude/longitude nan's and return filtered data
    data_sfr_without_nans = data_sfr.dropna(subset=['property_latitude', 'property_longitude'], how='any',axis=0)
    # Drop duplicates                                        
    data_sfr_without_nans.drop_duplicates(subset=['property_latitude', 'property_longitude', 'apn', 'gla_sqft', 'year'], keep='last', inplace=True, ignore_index=False)
    # Fill nan yearbuilt
    data_sfr_without_nans.year_built = np.where(data_sfr_without_nans.year_built.isna(), data_sfr_without_nans.groupby('city_correct').year_built.transform(lambda x: int(x.mean())), data_sfr_without_nans.year_built)  
                                        
    return data_sfr_without_nans

def filter_by_features(data_sfr):
    data_sfr = data_sfr[((data_sfr.bedrooms>2) & (data_sfr.bedrooms<8))] 
    data_sfr = data_sfr[((data_sfr.full_baths>0) & (data_sfr.full_baths<9))]  
    data_sfr = data_sfr.dropna(subset=['gla_sqft'], axis=0)
    data_sfr = data_sfr.loc[(data_sfr['price'] > 30000) & (data_sfr['price'] < 1000000)]
    data_sfr = data_sfr.loc[(data_sfr['price'] / data_sfr['gla_sqft']) > 15]
    
    return data_sfr

def prepare_data(data_sfr):
    '''categorical tranform '''
    numerical = ['gla_sqft', 'year_built','bedrooms', 'full_baths', 'property_longitude', 'property_latitude','city_correct','date']
    to_onehot = ['new_construction_flag'] # TODO add sale_type after changing to distressed/ fair market 
    transformed = ['fireplace','has_spa','bath_to_room','sqft_to_room','month','year', 'zip']
    data_sfr_ = data_sfr[numerical].copy()
    # categorical tranform
    data_sfr_['has_fireplace'] = data_sfr['fireplace_count'].fillna(0) > 0
    data_sfr_['has_spa'] = data_sfr['pool_spa_types'] == ''
    data_sfr_['bath_to_room'] = data_sfr.full_baths/data_sfr.bedrooms
    data_sfr_['sqft_to_room'] = data_sfr.gla_sqft/data_sfr.bedrooms
    data_sfr_['month'] = data_sfr.date.dt.month
    data_sfr_['year'] = data_sfr.date.dt.year
    data_sfr_['zip'] = data_sfr.zip.astype(int)
    
    encoder = OneHotEncoder().fit(data_sfr[to_onehot])
    transformed = encoder.transform(data_sfr[to_onehot]).toarray()
    transformed = pd.DataFrame(transformed, columns=encoder.get_feature_names(to_onehot), index=data_sfr.index)
    data_sfr_ = pd.concat([data_sfr_,transformed], axis=1)
    
    data_sfr['price_sqf'] = data_sfr.price/data_sfr.gla_sqft
    target = data_sfr[['price','price_sqf']].copy()
    return data_sfr_, target

class NaiveModel():

    def __init__(self):
        self.dist_km = 1
        self.n_comps = 5
        self.df = None
        self.comps = None
        self.rmse = None

    def filter_last_6_months(self, df, X):
        td = ((X['date'] - df['date']) / np.timedelta64(1, "M"))
        mask = (td < 6) & (td > 0)
        return df.loc[mask]

    def cut_by_area(self, df, X, area='city_correct'):
        return df.loc[df[area] == X[area]]

    def gross_filter(self, df, X):
        df_cut = self.cut_by_area(df, X)
        df_cut = self.filter_last_6_months(df_cut, X)
        return df_cut

    def fit(self, df):
        self.df = df

    def predict(self, X):
        df = self.gross_filter(self.df, X)
        neighbors = self.get_neighbors_by_distance(df, X)
        if neighbors is None:
            return None
        neighbors = neighbors.head(self.n_comps)
        return neighbors

    def pricing(self, X):
        comps = self.predict(X)
        try:
            ppsqft = int((comps['price'] / comps['gla_sqft']).mean())
            return ppsqft * X['gla_sqft']
        except:
            print("No comps found")
            return None

    def calc_rmse(self, X, y):
        #         if self.comps is None:
        comps = self.predict(X)
        self.rmse += np.sqrt(np.mean(np.square(comps['price'] - y))) / y
        return self.rmse

    def evaluate(self, year=2021, n_samples=100):
        if self.df is not None:
            sample = self.df[self.df['date'].dt.year == year].sample(n_samples, random_state=42)
            y = sample['price']
            self.rmse = 0
            Parallel(n_jobs=15, prefer="threads", verbose=0)(
                delayed(self.calc_rmse)(listing, y.loc[idx]) for idx, listing in tqdm(sample.iterrows()))
            #             for idx, listing in tqdm(sample.iterrows()):
            #                 rmse += self.calc_rmse(listing, y.loc[idx])
            self.rmse /= n_samples
            print(f"Model's rmse for year {year} and {n_samples} samples: {self.rmse:.5f}")
            return self.rmse
        else:
            print('Model has to be fit before evaluating')

    def build_distance_matrix(self, df, X):
        listing_coord = X[['property_latitude', 'property_longitude']]
        outher_coords = df[['property_latitude', 'property_longitude']]
        dists = cdist(listing_coord, outher_coords, lambda u, v: geodist(u, v).km)
        distance_matrix = pd.DataFrame(data=dists, columns=outher_coords.index, index=listing_coord.index)
        return distance_matrix

    def multi_cord_to_dist(self, df, X):
        '''
        Input:s X, a single listing to calculate distances from
                df, a dataframe of listings to calculate distances to

        Outputs: Dataframe containing distances (one-to-many)
        '''
        lat1 = X['property_latitude']
        lon1 = X['property_longitude']

        lat2 = df['property_latitude']
        lon2 = df['property_longitude']

        pi = np.pi / 180
        result = 0.5 - np.cos((lat2-lat1)*pi)/2 + np.cos(lat1*pi) * np.cos(lat2*pi) * (1-np.cos((lon2-lon1)*pi))/2
        result = 12742 * np.arcsin(np.sqrt(result))

        distance_matrix = pd.DataFrame(result, index=df.index, columns=[X.name])
        return distance_matrix.T
    
    def get_neighbors_by_distance(self, df, X):
        dist_km = self.dist_km
        dist_mat = self.multi_cord_to_dist(df, X)
        dist_mat.sort_values(by=dist_mat.iloc[0].name, axis='columns', ascending=True, inplace=True)
        self.dist_mat_km = dist_mat
        if ((dist_mat < dist_km).values).any():
            return df.loc[dist_mat.columns[(dist_mat < dist_km).values[0]]]
        else:
            return None

    def find_smallest_idx(self, distance_matrix):
        return distance_matrix.T.nsmallest(self.n_comps, distance_matrix.index).index
    
    def set_dist_km(self, dist_km):
        self.dist_km = dist_km
        
    def set_n_comps(self, n_comps):
        self.n_comps = n_comps
    

class BasicModel(NaiveModel):
    
    def __init__(self, weighted=False, weights=None):
        super().__init__()
        self.weighted = weighted
        if weights:
            multiplier = (len(weights) / sum(weights.values()))
            self.weights = {key: weights[key] * multiplier for key in weights.keys()}
#             self.weights = weights * multiplier
        elif weighted:
            self.weights = {}
    
    def dist_from_listing(self, df, X):
        dist_km = self.dist_km
        idx = self.dist_mat_km.columns[(self.dist_mat_km < dist_km).values[0]]
        idx_listing = self.dist_mat_km.index.values[0]
        df = df.merge(self.dist_mat_km.T.loc[idx], left_index=True, right_index=True, how='left')
        df.rename(columns={idx_listing:"dist_from_listing"}, inplace = True)
        X['dist_from_listing'] = 0
        return df
    
    def time_diff_from_listing(self, df, X):
        df['time_from_listing'] =  ((df['date'] - X['date']) / np.timedelta64(1, "M"))
        X['time_from_listing'] = 0
        return df
    
    def build_feature_distance_matrix(self, df, X):
#         dists = cdist(X, df, lambda u, v: distance.euclidean(u, v))
        dists = cdist(X, df, 'cosine')
        distance_matrix = pd.DataFrame(data=dists, index=X.index, columns=df.index)
        return distance_matrix
    

    def weights_by_LR(self, norm_df, target):
        norm_reg = LinearRegression().fit(norm_df, target)
        w = norm_reg.coef_
        return np.abs(w)/np.sum(np.abs(w))
    
    def normalize_features(self, df, X):
        weighted = self.weighted
        STANDARD_SCALING_COLS = ['bedrooms',
                                 'gla_sqft',
                                 'year_built',
                                 'full_baths',
                                 'property_longitude',
                                 'property_latitude',
                                 'has_fireplace',
                                 'has_spa',
                                 'bath_to_room',
                                 'sqft_to_room',
                                 'zip',
                                ]
        MAX_SCALING_COLS = ['time_from_listing',
                           'dist_from_listing']
        
        # TODO: Change weights after re-doing linear regression
        scaled_df = df[[*STANDARD_SCALING_COLS, *MAX_SCALING_COLS]].copy()
        scaled_X = X[[*STANDARD_SCALING_COLS, *MAX_SCALING_COLS]].copy()
        
        standard_scaler = StandardScaler()
        scaled_df[STANDARD_SCALING_COLS] = standard_scaler.fit_transform(scaled_df[STANDARD_SCALING_COLS]) #* weights
        scaled_X[STANDARD_SCALING_COLS] = standard_scaler.transform(scaled_X[STANDARD_SCALING_COLS])
        
        if weighted:
            weights = [self.weights[i] if i in self.weights else 1 for i in STANDARD_SCALING_COLS]
            
#             weights = self.weights_by_LR(scaled_df[STANDARD_SCALING_COLS], df.price)
            if len(weights) == len(STANDARD_SCALING_COLS):
                scaled_df[STANDARD_SCALING_COLS] *= weights
                scaled_X[STANDARD_SCALING_COLS] *= weights
            
        max_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_df[MAX_SCALING_COLS] = max_scaler.fit_transform(scaled_df[MAX_SCALING_COLS])*0.01
        scaled_X[MAX_SCALING_COLS] = max_scaler.transform(scaled_X[MAX_SCALING_COLS])*0.01
        
        return scaled_df, scaled_X
    
    def get_neighbors_by_features(self, df, X):
        dist_mat_feat = self.build_feature_distance_matrix(df, X)
        dist_mat_feat.sort_values(by=dist_mat_feat.iloc[0].name, axis='columns', ascending=True, inplace=True)
        return df.loc[dist_mat_feat.columns[:self.n_comps]]
    
    def predict(self, X):
        '''
        Finds n_comps comps for a given asset.
        Number of comps is setable by the set_n_comps function, while the default is 5 comps
        '''
        df = self.gross_filter(self.df, X)
#         print(df.shape[0], 'candidates')
        df = super().get_neighbors_by_distance(df, X)
        if df is None:
            return None
        if df.shape[0] == 0:
            print('Zero candidates after neighbors by distance')
            return None
#         print(df.shape[0], 'neighboring candidates around', X.name)
        df = self.dist_from_listing(df, X)
        if df.shape[0] == 0:
            print('Zero candidates after dist from listing')
            return None
        self.time_diff_from_listing(df, X)
        feat_df, feat_X = self.normalize_features(df, X.to_frame().T)
        
        neighbors = self.get_neighbors_by_features(feat_df, feat_X).index
        
        # Return the original dataframe rows corresponding to indices of neighbors
        return self.df.loc[neighbors]