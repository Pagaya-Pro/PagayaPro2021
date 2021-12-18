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
                 stratify_values=['co_mob', 'prepaid_mob'], sample_size=1, stratify=True):

        self.model_path = model_path
        self.cashflows = None
        self.predictions = None
        self.models = []

        if (type(feature_path) == type('')):
            try:
                target_data = parquet_to_dataframe(target_path)
            except:
                print('target data not split to folder - reading from parquet')
                target_data = pd.read_parquet(target_path)

            try:
                feature_data = parquet_to_dataframe(feature_path)
            except:
                print('feature data not split to folder - reading from parquet')
                feature_data = pd.read_parquet(feature_path)
        else:
            target_data = target_path
            feature_data = feature_path

        print('Creating cashflows')
        if (cashflows is None):
            cashflows = utils.survivals_to_cashflows(target_data)
            cashflows.set_index(target_data.index, inplace=True)
            cashflows['yearly_irr'] = self.calc_yearly_irr(cashflows)
            target_data['yearly_irr'] = cashflows['yearly_irr']
            print('Splitting data')
            if (cashflows_path is not None):
                self.cashflows_path = cashflows_path
                cashflows.to_parquet(cashflows_path)
            else:
                self.cashflows = cashflows
        else:
            cashflows.set_index(target_data.index, inplace=True)
            target_data['yearly_irr'] = cashflows['yearly_irr']
            self.cashflows = cashflows

        series_to_stratify = ([target_data[stratify_values[i]].astype(str) for i in range(len(stratify_values))])

        col_start = series_to_stratify[0]
        for col in series_to_stratify[1:]:
            col_start = col_start.str.cat(col, sep=',')

        target_data['stratify'] = col_start

        if stratify and sample_size < 1:
            _, feature_data, _, target_data = self.sample_data(sample_size, feature_data, target_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature_data, target_data['yearly_irr'],
                                                                                test_size=0.4, random_state=42,
                                                                                stratify=target_data['stratify'].fillna(
                                                                                    -1))
        print('done Splitting data')

    def sample_data(self, sample_size, feature_data, target_data):
        return train_test_split(feature_data, target_data, test_size=sample_size,
                                random_state=42, stratify=target_data['stratify'].fillna(-1))

    def add_zipcode_feature(self, func, col, zipcodes):
        self.X_train[col] = func(zipcodes.loc[X_train.sequence_num.values])
        self.X_test[col] = func(zipcodes.loc[X_test.sequence_num.values])

    def train_model(self, columns=[], num_boosts=10, save_model=False):
        if len(columns) == 0:
            columns = self.X_train.select_dtypes(include=np.number).columns

        print('Creating DMatrix')
        dtrain = xgb.DMatrix((self.X_train[columns]), self.y_train, nthread=-1)
        dtest = xgb.DMatrix((self.X_test[columns]), self.y_test, nthread=-1)
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

        num_boost_round = num_boosts

        print('Training model')
        model = xgb.train(params, dtrain, evals=[(dtest, 'eval'), (dtrain, 'train')], evals_result=evals_result,
                          num_boost_round=num_boost_round)
        self.dtrain = dtrain
        self.dtest = dtest
        self.model = model
        self.evals_result = evals_result
        self.predictions = model.predict(dtest)

        self.models.append((model, columns)) if save_model else print('Model not save for later comparison')

    def predict(self, X_test=None):

        dtest = None
        if type(X_test) == type(None):
            return (self.predictions)
        else:
            dtest = xgb.DMatrix(X_test, nthread=-1)

        return self.model.predict(dtest)

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

    def draw_curves(self, plot=True, regplot=True):

        if (self.cashflows is None):
            try:
                cashflows = pd.read_parquet(self.cashflows_path)
            except:
                print('No cashflows or cashflows file path not specified')
        else:
            cashflows = self.cashflows

        plt.figure(figsize=(20, 8))

        # Take model predictions
        self.X_test['predicted_model_cvs'] = self.predictions
        self.X_test.sort_values(by='predicted_model_cvs', ascending=False, inplace=True)
        # Calculate accumulated by rank
        self.X_test['accumulated_volume'] = self.X_test.loan_amnt.cumsum()

        # Get volumes and IRR values
        volumes, irrs = self.calc_volume_irr(self.X_test, cashflows, 'loan_amnt')

        # Change indeices back to original location
        self.X_test = self.X_test.loc[self.y_test.index]

        # Change indeices back to original location
        self.X_test.drop(columns=['predicted_model_cvs', 'accumulated_volume'], inplace=True)

        self.volumes, self.irrs = np.array(volumes), np.array(irrs)
        if plot:

            label = f'return on 50% volume - {str(np.round(irrs[9], 3))}'
            if regplot:
                sns.regplot(x=volumes / np.max(volumes), y=irrs, order=3, ci=None, label=label,
                            scatter_kws={"color": 'r'},
                            line_kws={"color": 'b'})
            else:
                plt.plot(volumes / np.max(volumes), irrs, label=label)

            plt.title('Volume vs IRR tradeoff')
            plt.legend(**{'title': 'Model'})
            plt.xlabel('Volume', size=12)
            plt.ylabel('IRR', size=12)
            plt.legend()
            plt.show()




