from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf
import os
import pandas as pd
import math
from file_utils import parquet_to_dataframe
import s3fs
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm
import psutil
import utils
import s3fs
import psutil
import pickle


class Train_model_pipeline():

    def __init__(self, feature_path, target_path, cashflows=None, cashflows_path=None, model_path=None,
                 stratify_values=['co_mob', 'prepaid_mob'], sample_size=1):

        self.model_path = model_path
        self.cashflows = None

        feature_data = parquet_to_dataframe(feature_path)
        target_data = parquet_to_dataframe(target_path)
        feature_data = self.add_features_from_target(feature_data, target_data)

        print('Creating cashflows')
        if (cashflows is None):
            cashflows = utils.survivals_to_cashflows(target_data)
            cashflows.set_index(target_data.index, inplace=True)
            target_data['yearly_irr'] = self.calc_yearly_irr(cashflows)
        else:
            cashflows.set_index(target_data.index, inplace=True)
            target_data['yearly_irr'] = cashflows['yearly_irr']

        print('Splitting data')
        if (cashflows_path is not None):
            self.cashflows_path = cashflows_path
            cashflows.to_parquet(cashflows_path)
        else:
            self.cashflows = cashflows

        series_to_stratify = ([target_data[stratify_values[i]].astype(str) for i in range(len(stratify_values))])

        col_start = series_to_stratify[0]
        for col in series_to_stratify[1:]:
            col_start = col_start.str.cat(col, sep=',')

        target_data['stratify'] = col_start

        if sample_size < 1:
            _, feature_data, _, target_data = self.sample_data(sample_size, feature_data, target_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature_data, target_data['yearly_irr'],
                                                                                test_size=0.33, random_state=42,
                                                                                stratify=target_data['stratify'].fillna(
                                                                                    -1))

    def sample_data(self, sample_size, feature_data, target_data):
        return train_test_split(feature_data, target_data, test_size=sample_size,
                                random_state=42, stratify=target_data['stratify'].fillna(-1))

    def add_zipcode_feature(self, func, col, zipcodes):
        self.X_train[col] = func(zipcodes.loc[X_train.sequence_num.values])
        self.X_test[col] = func(zipcodes.loc[X_test.sequence_num.values])

    def train_model(self):

        print('Creating DMatrix')
        self.dtrain = xgb.DMatrix(self.X_train.drop(columns=['sequence_num']), self.y_train, nthread=-1)
        self.dtest = xgb.DMatrix(self.X_test.drop(columns=['sequence_num']), self.y_test, nthread=-1)
        evals_result = dict()
        params = {
            "booster": "gbtree",
            "max_depth": 6,
            "colsample_bytree": 1,
            "subsample": 1,
            "eta": 0.1,
            "gamma": 1,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "nthread": 8,
            "seed": 42,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

        num_boost_round = 10

        print('Training model')
        self.model = xgb.train(params, self.dtrain, evals=[(self.dtest, 'eval'), (self.dtrain, 'train')],
                               evals_result=evals_result,
                               num_boost_round=num_boost_round)

        if (self.model_path is not None):
            with open(self.model_path, 'wb') as fp:
                pickle.dump(self.model, fp)

    def predict(self, X_test=None):
        if X_test is not None:
            dtest = xgb.DMatrix(X_test, nthread=-1)
        else:
            dtest = self.dtest

        return self.model.predict(dtest)

    def add_features_from_target(self, feature_data, target_data):
        try:
            feature_data['int_rate'] = target_data.int_rate
            feature_data['monthly_pmt'] = target_data.monthly_pmt
            feature_data['credit_score'] = target_data.credit_score
            feature_data['sequence_num'] = target_data.sequence_num
        except:
            print('Columns missing from train features.')

        return feature_data

    def calc_yearly_irr(self, cashflows):
        irrs = cashflows.swifter.apply(lambda x: npf.irr(x[x.notna()]), axis=1)
        irrs.fillna(-1, inplace=True)
        return (((irrs + 1) ** 12) - 1) * 100

    def plot_importance(self, num_features=15):
        ax = xgb.plot_importance(self.model, max_num_features=num_features)
        fig = ax.figure
        fig.set_size_inches(24, 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('F score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.show()

    def get_portfolio_irr(self, cashflows, indices, loan_amount_col='laon_amnt'):
        """
        cashflows- a dataframe of cashflows
        indices- pd.Series of T/F or a list of indices

        output: the yearly irr of the dataframe cashflows.loc[indices]
        """
        return (((npf.irr(cashflows.fillna(0).loc[indices].sum()) + 1) ** 12) - 1) * 100

    def calc_volume_irr(self, data, cashflows, loan_amount_col='loan_amnt'):
        volumes, irrs = [], []
        total_vol = data[loan_amount_col].sum()
        thresholds = np.linspace(0.05 * total_vol, total_vol, 20)
        for thresh in thresholds:
            portfolio_indices = data[data['accumulated_volume'] < thresh].index
            portfolio_irr = self.get_portfolio_irr(cashflows, portfolio_indices, loan_amount_col)

            volumes.append(thresh)
            irrs.append(portfolio_irr)
        return volumes, irrs

    def draw_curves(self):

        if (self.cashflows is None):
            try:
                cashflows = pd.read_parquet(self.cashflows_path)
            except:
                print('No cashflows or cashflows file path not specified')
        else:
            cashflows = self.cashflows

        plt.figure(figsize=(20, 8))
        # Copy dataset to avoid leakage of predicted columns to train/ test sets
        X_test_model = self.X_test.copy()
        # Take model predictions
        X_test_model['predicted_model_cvs'] = self.model.predict(self.dtest)
        X_test_model.sort_values(by='predicted_model_cvs', ascending=False, inplace=True)
        # Calculate accumulated by rank
        X_test_model['accumulated_volume'] = X_test_model.loan_amnt.cumsum()

        # Get volumes and IRR values
        volumes, irrs = self.calc_volume_irr(X_test_model, cashflows, 'loan_amnt')
        label = f'return on 50% volume - {str(np.round(irrs[9], 3))}'
        sns.regplot(x=volumes / np.max(volumes), y=irrs, order=3, ci=None, label=label,
                    scatter_kws={"color": 'r'},
                    line_kws={"color": 'b'})

        plt.title('Volume vs IRR tradeoff')
        plt.legend(**{'title': 'Model'})
        plt.xlabel('Volume', size=12)
        plt.ylabel('IRR', size=12)
        plt.legend()
        plt.show()

        self.volums = volumes
        self.irrs = irrs