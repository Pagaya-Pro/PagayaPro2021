import numpy as np
import pandas as pd
from Train_model_pipe import Train_model_pipeline
from file_utils import parquet_to_dataframe
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed


class Pca_Corr_Groups():

    def __init__(self, df, features_to_keep, corr_th, min_size_pca):

        self.df = df
        self.features_to_keep = features_to_keep
        self.corr_th = corr_th
        self.min_size_pca = min_size_pca

    def pca_the_df(self):

        original_num_of_cols = self.df.shape[1]
        self.pca = make_pipeline(StandardScaler(), PCA(n_components=1))
        self.pca_dict = {}
        print('Num of cols:', original_num_of_cols)
        print()

        for col in self.df.columns:

            if col not in self.df.columns:
                continue
            print('Working on column: ', col)
            print(' ')
            if col in cols_to_keep:
                print('col is a keeper')
                print(' ')
                continue

            _corr_group = self.df.corrwith(self.df[col]).abs()
            list__corr_group = list(_corr_group[_corr_group > self.corr_th].index)

            if len(list__corr_group) >= self.min_size_pca:
                for keeper in self.features_to_keep:
                    if keeper in list__corr_group:
                        list__corr_group.remove(keeper)
                        print('removed ' + keeper + ' from the list')
                        print(' ')

                print('creating test df')
                temp_df = self.df[list__corr_group].copy()
                _pca = self.pca.fit_transform(temp_df)
                del temp_df
                print('deleted test df')
                name_of_col = col + '_pca'
                self.pca_dict[name_of_col] = list__corr_group
                self.df[name_of_col] = _pca
                print(name_of_col + ' column created')
                self.df.drop(columns=list__corr_group, inplace=True)

                print(' ')
                print(len(list__corr_group), 'corr columns has been deleted from data frame')
                print(' ')
                print('End of circle')
                print('Num of cols:', self.df.shape[1])
                print(' ')
            else:
                print('Number of correalted features with', self.corr_th]., ' is les than', self.min_size_pca)
                print(' ')

                print(' ')
                print('HOLY MOLY!!! Process ended successfully!!!')
                return_string = 'from -> ' + str(original_num_of_cols) + 'to -> ' + str(
                    self.df.shape[1]) + ' number of columns'

        return return_string, self.pca_dict






