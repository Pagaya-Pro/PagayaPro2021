import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import swifter
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy import stats
from scipy.optimize import curve_fit
import pickle

import xgboost as xgb
from lifelines import CoxPHFitter

import warnings

warnings.simplefilter(action='ignore')

cols_to_table = ['follow_consumer_id', 'decline_reason', 'original_vantage_score', 'applicant_income', 'month',
                 'current_balance', 'principal_balance', 'balance_added', 'interest_added', 'minimum_payment_due',
                 'initial_apr', 'prime_rate', 'credit_limit', 'monthly_payments', 'monthly_payments_reversal',
                 'monthly_spend', 'monthly_disputes', 'monthly_cashback', 'dpd', 'charge_off_flag',
                 'hardship_principal_debits',
                 'card_status', 'loan_issue_date_full']

type_dict = {'original_vantage_score': float, 'applicant_income': float, 'current_balance': float,
             'principal_balance': float, 'balance_added': float, 'interest_added': float,
             'minimum_payment_due': float, 'initial_apr': float, 'prime_rate': float,
             'credit_limit': float, 'monthly_payments': float, 'monthly_payments_reversal': float,
             'monthly_spend': float, 'monthly_disputes': float, 'monthly_cashback': float,
             'dpd': float, 'hardship_principal_debits': float, 'loan_issue_date_full': 'datetime64[ns]'}

payments_features_cols = ['follow_consumer_id', 'original_vantage_score', 'applicant_income', 'initial_apr',
                          'charge_off_flag',
                          'loan_issue_date_full']

payments_cashflows_cols = ['current_balance', 'principal_balance', 'balance_added', 'interest_added',
                           'minimum_payment_due', 'prime_rate', 'credit_limit', 'monthly_payments',
                           'monthly_payments_reversal', 'monthly_spend', 'monthly_disputes', 'monthly_cashback',
                           'dpd', 'hardship_principal_debits', 'card_status']

xgboost_params = {
    "booster": "gbtree",
    "max_depth": 7,
    "colsample_bytree": 0.36,
    "subsample": 1,
    "eta": 0.06571,
    "gamma": 0.23285,
    "min_child_weight": 7,
    "max_delta_step": 0,
    "nthread": 8,
    "seed": 42,
    "objective": "reg:squarederror",
    "eval_metric": "rmse"
}
num_boost_round = 30

col_lst = [cols_to_table, type_dict, payments_features_cols, payments_cashflows_cols]


def read_data_base():
    print("reading database")
    since = time.time()
    features = pd.read_parquet('s3://pagaya-pro-bucket/projects/survival_analysis/petal_features.parquet')
    time_elapsed = time.time() - since
    print(f'Processing completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

    payments = pd.read_parquet('s3://pagaya-pro-bucket/projects/survival_analysis/petal_payments.parquet')

    return features, payments


# Processing payments

def missing_percentage(df):
    report = df.isna().sum().to_frame()
    report = report.rename(columns={0: 'missing_values'})
    report['per_of_total'] = (report['missing_values'] / df.shape[0]).round(4) * 100
    return report.sort_values(by='missing_values', ascending=False)


def get_intersection(features, payments):
    users_that_exists_in_payments_and_features = payments.groupby('follow_consumer_id')[
        'follow_consumer_id'].first().isin(features.sequence_num)
    users_that_exists_in_payments_and_features = users_that_exists_in_payments_and_features[
        users_that_exists_in_payments_and_features == True]
    payments = payments.loc[payments.follow_consumer_id.isin(users_that_exists_in_payments_and_features.index)]
    features = features.loc[features.sequence_num.isin(users_that_exists_in_payments_and_features.index)]
    return features, payments


def get_last_sequence(row, sequence_len=3):
    no_nan = row[~row.isna()]
    if (no_nan.size == 0):
        return 0
    idx_pairs = np.where(np.diff(np.hstack(([False], no_nan == 0, [False]))))[0].reshape(-1, 2)
    if (idx_pairs.size != 0) and (idx_pairs[-1][1] == len(no_nan)) and (np.diff(idx_pairs[-1])[0] >= sequence_len):
        return 1
    else:
        return 0


def get_mob(row, first_mob):
    return np.round((row['month'] - first_mob[row['follow_consumer_id']]) / np.timedelta64(1, 'M')).astype('int8')


def preprocess(features, payments, col_lst=col_lst, intersection=True, annoying_id=True,
               fill_credit_score=True, co_mob=True, ret_cols=False, last_seq_num=0, max_mob=True,
               last_seq_table=['monthly_payments'], add_agg=False, fill_na=False, count_status=False):
    since = time.time()
    payments_copy = payments.copy()
    features_copy = features.copy()

    print('Starting to process')

    if intersection:
        features_copy, payments_copy = get_intersection(features_copy, payments_copy)
    payments_copy = payments_copy[col_lst[0]]
    payments_copy.reset_index(drop=True, inplace=True)

    if 'decline_reason' in col_lst[0]:
        payments_copy = pd.get_dummies(data=payments_copy, columns=['decline_reason'])
        col_lst[2] += 'decline_reason_BAD_CASHFLOW_MODEL_SCORE', 'decline_reason_INTERNATIONAL_PEP_UNABLE_TO_VERIFY'

    payments_copy = payments_copy.astype(col_lst[1])
    if 'charge_off_flag' in col_lst[0]:
        payments_copy.charge_off_flag = np.where(payments_copy.charge_off_flag == 'TRUE', 1, 0)

    if 'month' in col_lst[0]:
        payments_copy.month = pd.to_datetime(payments_copy.month, format='%Y-%m-%d')
        first_mob = payments_copy.groupby('follow_consumer_id').month.min()
        payments_copy['mob'] = payments_copy.swifter.apply(lambda row: get_mob(row, first_mob), axis=1)
        payments_copy.sort_values(by=['follow_consumer_id', 'mob'], inplace=True)
        payments_copy.drop(columns='month', inplace=True)

    print('Building cashflows and features tables')

    agg_dict = {'card_status': (lambda val: val)}
    for num_col in col_lst[3][:-1]:
        agg_dict[num_col] = np.sum

    payments_cashflows = payments_copy.pivot_table(values=col_lst[3], index='follow_consumer_id', columns='mob',
                                                   aggfunc=agg_dict)
    payments_features = payments_copy[col_lst[2]].copy()
    payments_features.set_index('follow_consumer_id', drop=True, inplace=True)
    features_copy.set_index('sequence_num', drop=True, inplace=True)

    if annoying_id:
        mult_vals = payments_features.groupby('follow_consumer_id').applicant_income.unique().str.len() > 1
        ID = payments_features[payments_features.index == mult_vals[mult_vals].index[0]].index
        payments_features.drop(index=ID, inplace=True)
        payments_cashflows.drop(index=ID, inplace=True)
        features_copy.drop(index=ID, inplace=True)

    payments_features = payments_features[(~payments_features.index.duplicated())]
    features_copy = features_copy[(~features_copy.index.duplicated())]

    print()
    print('Filling missing values and adding additional features')

    if max_mob:
        payments_features['max_mob'] = (~payments_cashflows['monthly_payments'].isna()).sum(axis=1) - 1

    if fill_credit_score and ('original_vantage_score' in col_lst[0]):
        missing_credit_score = payments_copy[payments_copy.original_vantage_score.isna()].follow_consumer_id.unique()
        credit_score_from_features = (features_copy[features_copy.index.isin(missing_credit_score)]['credit_score'])
        credit_score_from_features = credit_score_from_features[~credit_score_from_features.index.duplicated()]

        mask = payments_features.original_vantage_score.isna()
        payments_features.loc[mask, 'original_vantage_score'] = payments_features.loc[
            mask, 'original_vantage_score'].fillna(credit_score_from_features)

    if co_mob:
        co_mob = payments_copy[payments_copy['charge_off_flag'] == 1].groupby('follow_consumer_id').mob.first()
        payments_features.loc[co_mob.index, 'charge_off_flag'] = co_mob
        payments_features.rename(columns={'charge_off_flag': 'co_mob'}, inplace=True)

    if last_seq_num:
        for table in last_seq_table:
            payments_cashflows[(table, '0_seq')] = payments_cashflows[table].swifter.apply(
                lambda row: get_last_sequence(row, sequence_len=last_seq_num), axis=1)

    if add_agg:
        for table in col_lst[3]:
            if table == 'card_status':
                continue
            payments_cashflows[(table, 'mean')] = payments_cashflows[table].mean(axis=1)
            payments_cashflows[(table, 'median')] = payments_cashflows[table].median(axis=1)
            payments_cashflows[(table, 'skew')] = payments_cashflows[table].skew(axis=1)
            payments_cashflows[(table, 'std')] = payments_cashflows[table].std(axis=1)

    if count_status:
        for i, status in enumerate(payments_copy.card_status.value_counts().index):
            if i == 0:
                payments_cashflows[('card_status', f'{status}_count')] = (
                        payments_cashflows['card_status'] == status).sum(axis=1)
            else:
                payments_cashflows[('card_status', f'{status}_count')] = (
                        payments_cashflows['card_status'].iloc[:, :-i] == status).sum(axis=1)

    if fill_na:
        payments_cashflows.fillna(0, inplace=True)

    time_elapsed = time.time() - since
    print()
    print(f'Processing completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

    if ret_cols:
        return payments_features, payments_cashflows, features_copy, col_lst[3]
    else:
        return payments_features, payments_cashflows, features_copy


def get_X_y(features, std_thres=0.1, corr_thres=0.8, random_state=42):
    missing_credit_indices = features[features.credit_score == 4].index
    have_credit_indices = features[features.credit_score > 4].index

    cols_std = features[features.index.isin(missing_credit_indices)].select_dtypes(include='number').std().fillna(
        0) > std_thres
    cols_std = cols_std[cols_std].index

    numeric_features = features[features.index.isin(have_credit_indices)].select_dtypes(include='number')
    cols_corr = numeric_features.corrwith(numeric_features.credit_score, axis=0).abs().sort_values(ascending=False)
    cols_corr = cols_corr[~cols_corr.isna()]
    cols_corr = cols_corr[cols_corr < corr_thres].index

    relevant_cols = set(cols_std).intersection(set(cols_corr))

    X = features[features.index.isin(have_credit_indices)][relevant_cols]
    y = features[features.index.isin(have_credit_indices)].credit_score
    real_X = features[features.index.isin(missing_credit_indices)][relevant_cols]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    return X, y, real_X


def final_processing(payments_features, payments_cashflows, features_copy, params,
                     num_boost_round, std_thres=0.1, corr_thres=0.8, random_state=42):
    payments_features_processed = payments_features.copy()
    payments_cashflows_processed = payments_cashflows.copy()
    features_processed = features_copy.copy()

    outliers_idx = payments_features[payments_features.applicant_income > 1000000].index

    X, y, real_X = get_X_y(features_copy, std_thres=std_thres, corr_thres=corr_thres, random_state=random_state)
    dtrain = xgb.DMatrix(X, y)
    dreal = xgb.DMatrix(real_X)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    credit_predictions = model.predict(dreal)
    real_X['credit_score'] = np.round(credit_predictions)
    fill_vals = real_X['credit_score']

    payments_features_processed.loc[
        payments_features_processed.original_vantage_score == 4, 'original_vantage_score'] = fill_vals
    features_processed.loc[features_processed.credit_score == 4, 'credit_score'] = fill_vals

    idx_INTERNATIONAL_PEP_UNABLE_TO_VERIFY = payments_features[(payments_features.iloc[:, -2] == 1)].index
    idx_BAD_CASHFLOW_MODEL_SCORE = payments_features[(payments_features.iloc[:, -1] == 1)].index
    declined_indices = np.concatenate((idx_INTERNATIONAL_PEP_UNABLE_TO_VERIFY, idx_BAD_CASHFLOW_MODEL_SCORE), axis=0)

    idx_to_remove = np.union1d(outliers_idx, declined_indices)
    payments_features_processed.drop(index=idx_to_remove, inplace=True)
    payments_cashflows_processed.drop(index=idx_to_remove, inplace=True)
    features_processed.drop(index=idx_to_remove, inplace=True)

    payments_features_processed.drop(columns=['decline_reason_BAD_CASHFLOW_MODEL_SCORE',
                                              'decline_reason_INTERNATIONAL_PEP_UNABLE_TO_VERIFY'], inplace=True)
    payments_features_processed['co_indicator'] = payments_features_processed.co_mob > 0

    return payments_features_processed, payments_cashflows_processed, features_processed


def get_label_sklearn(payments, cut=None, add_one=True, change_type=False, vertical=False):
    y = payments[['co_indicator', 'co_mob', 'max_mob']].copy()
    y.loc[y['co_mob'] > 0, 'max_mob'] = y.loc[y['co_mob'] > 0, 'co_mob']

    if cut is not None:
        if vertical:
            y = y[y.max_mob > cut]
            y.max_mob -= cut
        else:
            y.loc[y['max_mob'] > cut, 'co_indicator'] = False
            y.loc[y['max_mob'] > cut, 'max_mob'] = cut
            if add_one:
                y.max_mob += 1

    y.drop(columns='co_mob', inplace=True)
    if change_type:
        y = y.swifter.apply(lambda row: (row.co_indicator, row.max_mob), axis=1).values.astype(
            [('co_indicator', '?'), ('max_mob', '<f8')])
    return y


# Vanilla model

def update_label(X, y):
    X = X.loc[y.index].copy()
    X.max_mob = y.max_mob
    X.co_indicator = y.co_indicator
    return X


def vertical_cut(payments, X, cut=12):
    assert X.max_mob.max() > cut, f'cut must be smaller than max_mob maximal value'

    y_train = get_label_sklearn(payments, cut=cut)
    y_test = get_label_sklearn(payments, cut=cut, vertical=True)
    X_train = update_label(X, y_train)
    X_test = update_label(X, y_test)
    X_test.max_mob += cut

    return X_train, X_test


def vanilla_process(features, payments, encode_state=True, PCA_components=50, present_explained_variance=True,
                    drop_nan=True, nan_per=0.6, random_state=4, label_cut=None,
                    include_y_in_X=True, test_size=0.33, standardize=True, split=True):
    X = features.copy()
    y = get_label_sklearn(payments, cut=label_cut)
    X = pd.concat([X, y], axis=1)

    scores_cols = X.filter(like='score').columns.drop('finscore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if encode_state:
        state_encoding = X_train.groupby('state_tu').co_indicator.mean()
        X.state_tu = X.state_tu.map(state_encoding)
        scores_cols = np.insert(scores_cols, 0, 'state_tu')

    X_keep = X[scores_cols]
    X_keep.fillna(X_keep.loc[X_train.index].mean(axis=0), axis=0, inplace=True)
    X_numeric = X.select_dtypes(include='number').drop(columns=scores_cols).iloc[:, :-2]

    if drop_nan:
        drop_cols = X_numeric.loc[X_train.index].isna().sum() > (nan_per * len(X_train))
        X_numeric.drop(columns=drop_cols[drop_cols].index, inplace=True)
        X_numeric.fillna(X_numeric.loc[X_train.index].mean(axis=0), axis=0, inplace=True)

    std_sclr = StandardScaler()
    pca = PCA(PCA_components, random_state=random_state)

    if split:
        std_sclr.fit(X_numeric.loc[X_train.index])
        pca.fit(std_sclr.transform(X_numeric.loc[X_train.index]))
    else:
        std_sclr.fit(X_numeric)
        pca.fit(std_sclr.transform(X_numeric))

    features_pca = pd.DataFrame(pca.transform(std_sclr.transform(X_numeric)), index=X_numeric.index)

    if standardize:
        if split:
            std_sclr.fit(X_keep.loc[X_train.index])
        else:
            std_sclr.fit(X_keep)

        X_keep = pd.DataFrame(std_sclr.transform(X_keep), index=X_keep.index, columns=X_keep.columns)

    if present_explained_variance:
        pca_explained_variance = sum(pca.explained_variance_ratio_)
        print(f'Processing completed, explained variance for {PCA_components} components is {pca_explained_variance}')

    if include_y_in_X:
        X = pd.concat([features_pca, X_keep, y], axis=1)
        if split:
            return X.loc[X_train.index], X.loc[X_test.index]
        else:
            return X
    else:
        X = pd.concat([features_pca, X_keep], axis=1)
        if split:
            return X.loc[X_train.index], X.loc[X_test.index], y_train, y_test
        else:
            return X, y


def remove_correlation(X_train, X_test=None, include_y=True, present_progress=True, remove_amnt=2,
                       PCA_components=50, print_col_names=True):
    if include_y:
        correlation_matrix = X_train.corr().iloc[PCA_components:-2, PCA_components:-2]
    else:
        correlation_matrix = X_train.corr().iloc[PCA_components:, PCA_components:]

    if present_progress:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix)
        plt.title('correlation matrix before filtering', fontsize=15)
        plt.show()
        print()

    remove_cols = correlation_matrix.abs().sum().sort_values(ascending=False)[:remove_amnt].index

    if print_col_names:
        print(f'Removed columns are:')
        for i in range(remove_amnt):
            print(f'{remove_cols[i]}')
        print()

    if present_progress:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix.drop(index=remove_cols, columns=remove_cols))
        plt.title('correlation matrix after filtering', fontsize=15)
        plt.show()

    if X_test is None:
        return X_train.drop(columns=remove_cols)
    else:
        return X_train.drop(columns=remove_cols), X_test.drop(columns=remove_cols)


# V1 - payments enriched model

def get_features_from_payments(payments_cashflows, payments, cut_months=None):
    '''
    payments_cashflows: cashflows tables that Oded created.
    payments: payments table from tu.
    cut_month: np array of months we would like to consider.
    '''
    if cut_months is None:
        cut_months = payments_features_processed.max_mob.max()

    df = pd.DataFrame(index=payments_cashflows.index)

    df['balance_mean'] = payments_cashflows['current_balance'].loc[:, :cut_months].mean(1)
    df['credit_limit'] = payments_cashflows['credit_limit'].loc[:, :cut_months].mean(1)
    df['monthly_disputes'] = payments_cashflows['monthly_disputes'].loc[:, :cut_months].mean(1)
    df['max_dpd'] = payments_cashflows['dpd'].loc[:, :cut_months].max(1)
    df['status_not_B'] = (payments_cashflows_processed['card_status'].loc[:, :cut_months] != 'B').sum(axis=1)
    df.status_not_B.fillna(0, inplace=True)

    return df


def feature_engineering(pc, features, cut_months=None):
    pc = pc.copy()
    df = pd.DataFrame(index=payments_cashflows.index)

    if cut_months is None:
        cut_months = features.max_mob.max()
    cols = ['prime_rate']
    min_pmt = pc.minimum_payment_due.loc[:, 1:cut_months]
    monthly_pmt = pc.monthly_payments.loc[:, :cut_months - 1]
    min_pmt.columns -= 1
    #     display(min_pmt)
    #     display(monthly_pmt)

    indicator = (min_pmt <= monthly_pmt).sum(axis=1) / cut_months
    df['pay_bigger_min_pay'] = indicator
    df['new_fico'] = features.original_vantage_score
    #     df['init_rate'] = payments_features_processed.prime_rate
    #     balance = pc.principal_balance.loc[:, :cut_months]
    #     balance.colums = 1

    #     res = None
    #     for i in range(balance.shape[1]):
    #         temp = balance.iloc[:, :i] > balance.iloc[:, :i+1]
    #         if res == None:
    #             res = temp
    #         else:
    #             res = res & temp
    #     df['balance_get_down'] = res
    return df


def get_decreasing_balance_indicator(row, sequence_len=3):
    no_nan = row[~row.isna()].values
    if (no_nan.size <= sequence_len):
        return False
    return (no_nan[-(sequence_len - 1):] < no_nan[-sequence_len:-1]).all()


def vanilla_process(features, payments, encode_state=True, PCA_components=50, present_explained_variance=True,
                    drop_nan=True, nan_per=0.6, random_state=4, label_cut=None,
                    include_y_in_X=True, test_size=0.33, standardize=True, split=True):
    X = features.copy()
    y = get_label_sklearn(payments, cut=label_cut)
    X = pd.concat([X, y], axis=1)

    numeric_features = X.select_dtypes(include='number')
    only_one_value_under_0 = numeric_features.columns[numeric_features[numeric_features < 0].std() == 0]
    for col in only_one_value_under_0:
        numeric_features.loc[numeric_features[col] >= 0, col].mean()
        numeric_features[col].replace(numeric_features[col][numeric_features[col] < 0].unique()[0],
                                      numeric_features.loc[numeric_features[col] >= 0, col].mean(), inplace=True)

    scores_cols = X.filter(like='score').columns.drop('finscore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if encode_state:
        state_encoding = X_train.groupby('state_tu').co_indicator.mean()
        X.state_tu = X.state_tu.map(state_encoding)
        scores_cols = np.insert(scores_cols, 0, 'state_tu')

    X_keep = X[scores_cols]
    X_keep.fillna(X_keep.loc[X_train.index].mean(axis=0), axis=0, inplace=True)
    X_numeric = X.select_dtypes(include='number').drop(columns=scores_cols).iloc[:, :-2]

    if drop_nan:
        drop_cols = X_numeric.loc[X_train.index].isna().sum() > (nan_per * len(X_train))
        X_numeric.drop(columns=drop_cols[drop_cols].index, inplace=True)
        X_numeric.fillna(X_numeric.loc[X_train.index].mean(axis=0), axis=0, inplace=True)

    std_sclr = StandardScaler()
    pca = PCA(PCA_components, random_state=random_state)

    if split:
        std_sclr.fit(X_numeric.loc[X_train.index])
        pca.fit(std_sclr.transform(X_numeric.loc[X_train.index]))
    else:
        std_sclr.fit(X_numeric)
        pca.fit(std_sclr.transform(X_numeric))

    features_pca = pd.DataFrame(pca.transform(std_sclr.transform(X_numeric)), index=X_numeric.index)

    if standardize:
        if split:
            std_sclr.fit(X_keep.loc[X_train.index])
        else:
            std_sclr.fit(X_keep)

        X_keep = pd.DataFrame(std_sclr.transform(X_keep), index=X_keep.index, columns=X_keep.columns)

    if present_explained_variance:
        pca_explained_variance = sum(pca.explained_variance_ratio_)
        print(f'Processing completed, explained variance for {PCA_components} components is {pca_explained_variance}')

    if include_y_in_X:
        X = pd.concat([features_pca, X_keep, y], axis=1)
        if split:
            return X.loc[X_train.index], X.loc[X_test.index]
        else:
            return X
    else:
        X = pd.concat([features_pca, X_keep], axis=1)
        if split:
            return X.loc[X_train.index], X.loc[X_test.index], y_train, y_test
        else:
            return X, y


# Creating a correct baseline hazard $H_0(t)$

from scipy.special import gamma


def func(x, a, b):
    return a * np.exp(-b * x)


def func2(x, a, b):
    return (b ** a) * (np.e ** (-b * x) * (x ** (a - 1))) / gamma(a)


def create_basic_h_0(label):
    co_per_month = label.loc[label['co_indicator'], 'max_mob'].value_counts().sort_index()
    index_set = set(co_per_month.index)
    for i in range(1, co_per_month.index.max() + 1):
        if i not in index_set:
            co_per_month[i] = 0

    co_per_month.sort_index(inplace=True)
    cumsum = (len(label) - np.cumsum(co_per_month))[:-1]
    cumsum[0] = cumsum[1]
    cumsum.index += 1
    cumsum.sort_index(inplace=True)
    return co_per_month / cumsum


def get_h_0(label, n_step, n_month, month_for_train=None):
    '''
    label: the target label
    n_step: the number of month to calculate h0 as step function
    n_month: the number of month the function predict
    month_for_train: the number of month from n_step that the exponential regression train on
    '''

    step_h_0 = create_basic_h_0(label.copy())

    predict = step_h_0[:n_step].copy()
    y_train = step_h_0[n_step:].copy()
    if month_for_train is not None:
        y_train = step_h_0[n_step:n_step + month_for_train]

    popt, pcov = curve_fit(func, y_train.index, y_train)
    alpha, beta = popt[0], popt[1]
    y_predict = func(np.arange(n_step, n_month), alpha, beta)

    for i in range(n_step, n_month):
        predict[i] = y_predict[i - n_step]

    return alpha, -beta, predict, step_h_0


# alpha, beta, predict, step_h_0 = get_h_0(X_train.iloc[:, -2:], 12, 36)
#
# plt.figure(figsize=(16, 6))
# plt.plot(step_h_0, "-o", label='True baseline hazard')
# plt.plot(predict, label='Exponential H0')
# plt.title('Step H0 vs to estimated exponential hazard', fontsize=15)
# plt.legend()
# plt.xlabel('mob')
# plt.ylabel('density')
# plt.show()
#
# # Evaluation
#
# model_h_0 = pd.Series(cph.baseline_hazard_.to_numpy().reshape(-1, ) * (1 / 0.72), index=np.arange(2, 30))
# enriched_model_h_0 = pd.Series(cph_enriched.baseline_hazard_.to_numpy().reshape(-1, ) * (1 / 0.72),
#                                index=np.arange(2, 30))
# _, _, expon_h_0, train_h_0 = get_h_0(X_train.iloc[:, -2:], 12, 36)
# test_h_0 = create_basic_h_0(X_test.iloc[:, -2:])
#
# plt.figure(figsize=(16, 6))
#
# plt.plot(train_h_0, "o", label='True train baseline hazard')
# plt.plot(test_h_0, "o", label='True test baseline hazard')
# plt.plot(model_h_0, "-", label='Vanilla model baseline hazard')
# plt.plot(enriched_model_h_0, "-", label='Enriched model baseline hazard')
#
# plt.plot(expon_h_0, label='Exponential H0')
#
# plt.legend()
# plt.title("Baseline hazard distribution comparison\n Model's calculated function to true proportions", fontsize=15)
# plt.xlabel('mob')
# plt.show()


def get_density_stats(data1, data2, alpha=0.05, print_quartiles=True):
    density_data1, density_data2 = (data1 / np.trapz(data1)), (data2 / np.trapz(data2))
    min_mob, max_mob = min(data1.index.min(), data1.index.min()), max(data1.index.max(), data1.index.max())
    density_min = np.zeros(max_mob - min_mob + 1)

    for mob in range(min_mob, max_mob + 1):
        if mob in data1.index and mob in data2.index:
            density_min[mob - min_mob] = min(density_data1[mob], density_data2[mob])

    intersection = np.trapz(density_min)
    if intersection < (1 - alpha):
        print(
            f'intersection integral is {intersection:.3f}, which is smaller than {(1 - alpha) * 100:.0f}% percent of our probability functions.\nWe therefore reject the null hypothesis.')
    else:
        print(
            f'intersection integral is {intersection:.3f}, which is greater than {(1 - alpha) * 100:.0f}% percent of our probability functions.\nWe therefore cannot reject the null hypothesis.')

    if print_quartiles:
        print()

        plt.figure(figsize=(16, 6))
        plt.plot(data1, "-", label='First input')
        plt.plot(data2, "-", label='Second input')
        plt.legend()
        plt.title("Probability functions, described by quartiles", fontsize=15)
        plt.xlabel('mob')
        plt.show()


# get_density_stats(model_h_0, expon_h_0)


def fabricate_distribution(prob_func):
    mult = 1000
    baseline_distribution = (prob_func * mult).astype(int)
    while np.isin(0, baseline_distribution[prob_func > 0]):
        mult *= 10
        baseline_distribution = (prob_func * mult).astype(int)

    fabricated_sample = np.ones(1)
    for i in range(len(prob_func)):
        if i in baseline_distribution.index:
            fabricated_sample = np.append(fabricated_sample, np.ones(baseline_distribution[i]) * i)
    fabricated_sample = fabricated_sample[1:] + 1

    return fabricated_sample


significant_level = {0: '25', 1: '10', 2: '5', 3: '2.5', 4: '1', 5: '0.5', 6: '0.1'}


def get_stats_by_fabrication(prob_func, label, alpha=0.05, print_quartiles=True):
    true_baseline = create_basic_h_0(label)
    fabricated_sample = fabricate_distribution(prob_func[:label.max_mob.max()])
    statistic, critical_values, significance_level = stats.anderson_ksamp([fabricated_sample,
                                                                           label[label.co_indicator].max_mob.values])
    print(statistic)
    print(critical_values)
    significancy = (critical_values < statistic).sum() - 1
    if significancy == -1:
        print('We cannot reject the null hypothesis')
    else:
        print(f'We can reject the null hypothesis under {significant_level[significancy]}% significancy')

    if print_quartiles:
        print()

        plt.figure(figsize=(16, 6))
        plt.plot(prob_func[:label.max_mob.max()], "-", label='First input')
        plt.plot(true_baseline, "-", label='Second input')
        plt.legend()
        plt.title("Probability functions, described by quartiles", fontsize=15)
        plt.xlabel('mob')
        plt.show()


def predict_hazard_function(model, sample, h_0, times):
    '''
    model: lifelines cox hazard model after fit
    samples: dataframe of features of the sample which we want to predict on
    h_0: Series of our new h_0.
    times: Series/np.array of times to predict
    '''
    partial_hazard = model.predict_partial_hazard(sample)
    results = pd.DataFrame(columns=sample.index)
    for i in times:
        results.loc[i] = (h_0.loc[i] * partial_hazard)
    return results


def find_hazard_constant(cph, X_train, show_graph=False):
    # label the train
    groups_by_co = co_pop(X_train)
    start = groups_by_co.index[0]
    end = groups_by_co.index[-1]
    # get h0 for train
    alpha, beta, baseline, step_h_0 = get_h_0(X_train.iloc[:, -2:], 11, end + 1)
    baseline = baseline.loc[start:]
    predict_hazard = predict_hazard_function(cph, X_train, baseline, np.arange(start, end + 1)).mean(axis=1)

    y_actual = groups_by_co
    # find the constant:
    if show_graph:
        (predict_hazard / y_actual).plot()
        plt.title('ratio between predict hazard and label real population')
        plt.xlabel('month')
        plt.ylabel('ratio')
    hazard_constant = 1 / (predict_hazard / y_actual).mean()
    return hazard_constant


def basline_and_co_pop(X_train, X_test, start_exp=8, split_horizontal=True, title=""):
    from sklearn.metrics import mean_squared_error
    import math
    groups_by_co = co_pop(X_test)
    start_test = groups_by_co.index[0]
    end_test = groups_by_co.index[-1]
    alpha, beta, baseline, step_h_0 = get_h_0(X_train.iloc[:, -2:], 11, end_test + 1)

    baseline = baseline[start_test - 1:]
    predict_hazard = predict_hazard_function(cph, X_test, baseline, np.arange(start_test, end_test + 1)).mean(axis=1)

    y_actual = groups_by_co
    # find the constant:
    hazard_constant = 1 / cph.predict_partial_hazard(X_train).mean()
    y_predicted = predict_hazard * hazard_constant

    sns.set_style("whitegrid")
    sns.lineplot(data=y_actual, label="CO(t)/POP(t)")
    sns.lineplot(data=y_predicted, label="predicted hazard function")
    plt.xlabel("CO Month")
    plt.ylabel("probability")

    plt.title(title)
    plt.legend()
    plt.show()

    MSE = mean_squared_error(y_actual, y_predicted)
    RMSE = math.sqrt(MSE)
    print("RMSE test between actual and hazard function: ", RMSE)


def co_pop(X):
    co_mob_data = X[X.co_indicator == True][['max_mob']]

    groups_by_co = co_mob_data.groupby('max_mob').max_mob.count()

    size_of_population = len(X)
    for i in range(X.max_mob.min(), X.max_mob.max() + 1):
        if i in groups_by_co.index:
            num_of_co = groups_by_co.loc[i]
            groups_by_co.loc[i] = num_of_co / size_of_population
            size_of_population -= num_of_co
        else:
            groups_by_co.loc[i] = 0
    return groups_by_co


def eval_test(cph):
    # fit on X predict on X
    X = vanilla_process(features_processed, payments_features_processed, split=False)
    X = remove_correlation(X, remove_amnt=3, present_progress=False)
    cph.fit(X, 'max_mob', event_col='co_indicator', step_size=0.4)
    basline_and_co_pop(X, X, title='fitted and test on the entire data')

    # fit on train, predict on test-horizontal
    X_train, X_test = vanilla_process(features_processed, payments_features_processed, test_size=0.4)
    X_train, X_test = remove_correlation(X_train, X_test, remove_amnt=3, present_progress=False)
    cph.fit(X_train, 'max_mob', event_col='co_indicator', step_size=0.4)
    basline_and_co_pop(X_train, X_train, title='fitted on train(horizontal) and test on the train set(horizontal)')
    basline_and_co_pop(X_train, X_test, title='fitted on train(horizontal) and test on the test set(horizontal)')

    # fit on train, predict on test-vertical
    X_train_vertical, X_test_vertical = vertical_cut(payments_features_processed, X, cut=18)
    X_train_vertical, X_test_vertical = remove_correlation(X_train_vertical, X_test_vertical, remove_amnt=3,
                                                           present_progress=False)
    cph.fit(X_train_vertical, 'max_mob', event_col='co_indicator', step_size=0.4)
    basline_and_co_pop(X_train_vertical, X_test_vertical, split_horizontal=False,
                       title='fitted on train(vertical) and test on the test set(vertical)')

    # fit on train-horizontal and vertical, predict on test-horizontal and vertical
    X_train_first, X_train_second, _, _ = train_test_split(X_train_vertical, X_train_vertical.iloc[:, -2:],
                                                           test_size=0.4, random_state=4)
    X_test_first, X_test_second, _, _ = train_test_split(X_test_vertical, X_test_vertical.iloc[:, -2:], test_size=0.4,
                                                         random_state=4)
    cph.fit(X_train_first, 'max_mob', event_col='co_indicator', step_size=0.4)
    basline_and_co_pop(X_train_first, X_test_second, split_horizontal=False,
                       title='fitted on train(vertical and horizental) and test on the test set(vertical and horizental)')

    # fit on train-horizontal and vertical, predict on test-horizontal
    basline_and_co_pop(X_train_first, X_test, split_horizontal=False,
                       title='fitted on train(vertical and horizental) and test on the test set')


# V2 - better feature selection
def final_processing(payments_features, payments_cashflows, features_copy, params=xgboost_params,
                     num_boost_round=num_boost_round, std_thres=0.1, corr_thres=0.8, random_state=42):
    payments_features_processed = payments_features.copy()
    payments_cashflows_processed = payments_cashflows.copy()
    features_processed = features_copy.copy()

    outliers_idx = payments_features[payments_features.applicant_income > 1000000].index

    X, y, real_X = get_X_y(features_copy, std_thres=std_thres, corr_thres=corr_thres, random_state=random_state)
    dtrain = xgb.DMatrix(X, y)
    dreal = xgb.DMatrix(real_X)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    credit_predictions = model.predict(dreal)
    real_X['credit_score'] = np.round(credit_predictions)
    fill_vals = real_X['credit_score']

    payments_features_processed.loc[
        payments_features_processed.original_vantage_score == 4, 'original_vantage_score'] = fill_vals
    features_processed.loc[features_processed.credit_score == 4, 'credit_score'] = fill_vals

    idx_INTERNATIONAL_PEP_UNABLE_TO_VERIFY = payments_features[(payments_features.iloc[:, -2] == 1)].index
    idx_BAD_CASHFLOW_MODEL_SCORE = payments_features[(payments_features.iloc[:, -1] == 1)].index
    declined_indices = np.concatenate((idx_INTERNATIONAL_PEP_UNABLE_TO_VERIFY, idx_BAD_CASHFLOW_MODEL_SCORE), axis=0)

    idx_to_remove = np.union1d(outliers_idx, declined_indices)
    payments_features_processed.drop(index=idx_to_remove, inplace=True)
    payments_cashflows_processed.drop(index=idx_to_remove, inplace=True)
    features_processed.drop(index=idx_to_remove, inplace=True)

    payments_features_processed.drop(columns=['decline_reason_BAD_CASHFLOW_MODEL_SCORE',
                                              'decline_reason_INTERNATIONAL_PEP_UNABLE_TO_VERIFY'], inplace=True)
    payments_features_processed['co_indicator'] = payments_features_processed.co_mob > 0

    return payments_features_processed, payments_cashflows_processed, features_processed


def get_decreasing_balance_indicator(row, sequence_len=3):
    no_nan = row[~row.isna()].values
    if (no_nan.size <= sequence_len):
        return False
    return (no_nan[-(sequence_len - 1):] < no_nan[-sequence_len:-1]).all()


def is_only_9(num):
    return str(int(num)).count('9') == len(str(int(num)))


def enrich_from_payments_really(payments_cashflows, payments_features, cut_months=None, sequence_len=3):
    '''
    payments_cashflows: cashflows tables that Oded created.
    payments: payments table from tu.
    cut_month: np array of months we would like to consider.
    '''
    if cut_months is None:
        cut_months = payments_features.max_mob.max()
    min_between_cut_month_and_max_mob = payments_features.max_mob.swifter.apply(
        lambda max_mob: min(cut_months, max_mob))
    df = pd.DataFrame(index=payments_cashflows.index)

    df['status_is_B'] = (payments_cashflows['card_status'].loc[:, :cut_months] == 'B').sum(
        axis=1) / min_between_cut_month_and_max_mob
    df['status_is_A'] = (payments_cashflows['card_status'].loc[:, :cut_months] == 'A').sum(
        axis=1) / min_between_cut_month_and_max_mob
    df['status_is_C_or_I'] = ((payments_cashflows['card_status'].loc[:, :cut_months] == 'C').sum(axis=1) \
                              + (payments_cashflows['card_status'].loc[:, :cut_months] == 'I').sum(
                axis=1)) / min_between_cut_month_and_max_mob
    df['mean_balance_ratio'] = payments_cashflows.monthly_payments.loc[:, :cut_months].mean(
        1) + payments_cashflows.current_balance.loc[:, :cut_months].mean(1)
    df['usage'] = (payments_cashflows['monthly_spend'].loc[:, :cut_months] < 0).sum(
        axis=1) / min_between_cut_month_and_max_mob
    df['decreasing_balance'] = payments_cashflows['current_balance'].loc[:, :cut_months].swifter.apply(
        lambda row: get_decreasing_balance_indicator(row, sequence_len), axis=1)
    df['initial_apr'] = payments_features.initial_apr
    min_pmt = payments_cashflows.minimum_payment_due.loc[:, 1:cut_months] - 1
    monthly_pmt = payments_cashflows.monthly_payments.loc[:, :cut_months - 1]
    min_pmt.columns -= 1
    df['pay_bigger_min_pay'] = (min_pmt <= monthly_pmt).sum(axis=1) / cut_months

    return df


def best_model_process(features, payments, payments_cashflows, random_state=4, label_cut=None, test_size=0.20,
                       standardize=True, split=True):
    X = features.copy()
    y = get_label_sklearn(payments, cut=label_cut)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    numeric_features = X.select_dtypes(include='number').copy()
    numeric_features_train = X_train.select_dtypes(include='number').copy()
    only_one_value_under_0 = numeric_features.columns[numeric_features_train[numeric_features_train < 0].std() == 0]
    if split:
        for col in only_one_value_under_0:
            numeric_features[col].replace(numeric_features[col][numeric_features[col] < 0].unique()[0],
                                          numeric_features_train.loc[numeric_features_train[col] >= 0, col].mean(),
                                          inplace=True)

        # remove large numbers with only 9 digit
        to_drop = [col for col in numeric_features_train.columns if
                   numeric_features_train[col].value_counts().shape[0] < 2]
        to_drop_temp = [col for col in numeric_features_train.columns if
                        numeric_features_train[col].value_counts().shape[0] < 3]
        features_mor_then_2 = numeric_features_train.drop(to_drop_temp, axis=1)
        features_processed_max = features_mor_then_2.max()
        features_processed_2_max = features_mor_then_2.replace(features_mor_then_2.max(),
                                                               features_mor_then_2.min()).max()
        features_processed_min = features_mor_then_2.min()
        haw_far = (features_processed_max - features_processed_2_max) / (
                features_processed_2_max - features_processed_min)
        features_mor_then_2.loc[:, (haw_far < 1) & (features_processed_max.apply(is_only_9))].replace(
            features_mor_then_2.max(), np.nan, inplace=True)
        numeric_features[features_mor_then_2.columns] = features_mor_then_2
        numeric_features.fillna(numeric_features.mean())
        X[numeric_features.columns] = numeric_features
        X.drop(to_drop, axis=1, inplace=True)
        # handling categories
        to_remove = []
        for col in X_train.select_dtypes(include=['object']).columns:
            if (X_train[col].value_counts().shape[0] < 2) or (X_train[col].value_counts()[0] < 10):
                to_remove.append(col)
            else:
                col_to_co_percentage = X_train.join(y).groupby(col).co_indicator.mean()
                X[col] = X[col].map(col_to_co_percentage)
        X.drop(to_remove, axis=1, inplace=True)
        with_month = X_train.corrwith(y.loc[X_train.index][y.loc[X_train.index].co_indicator].max_mob).sort_values(
            key=lambda x: np.abs(x), ascending=False)
        with_co = X_train.corrwith(y.loc[X_train.index].co_indicator).sort_values(key=lambda x: np.abs(x),
                                                                                  ascending=False)

    else:
        for col in only_one_value_under_0:
            numeric_features[col].replace(numeric_features[col][numeric_features[col] < 0].unique()[0],
                                          numeric_features.loc[numeric_features[col] >= 0, col].mean(), inplace=True)

        # remove large numbers with only 9 digit
        to_drop = [col for col in numeric_features.columns if numeric_features[col].value_counts().shape[0] < 2]
        to_drop_temp = [col for col in numeric_features.columns if numeric_features[col].value_counts().shape[0] < 3]
        features_mor_then_2 = numeric_features.drop(to_drop_temp, axis=1)
        features_processed_max = features_mor_then_2.max()
        features_processed_2_max = features_mor_then_2.replace(features_mor_then_2.max(),
                                                               features_mor_then_2.min()).max()
        features_processed_min = features_mor_then_2.min()
        haw_far = (features_processed_max - features_processed_2_max) / (
                features_processed_2_max - features_processed_min)
        features_mor_then_2.loc[:, (haw_far < 1) & (features_processed_max.apply(is_only_9))].replace(
            features_mor_then_2.max(), np.nan, inplace=True)
        numeric_features[features_mor_then_2.columns] = features_mor_then_2
        numeric_features.fillna(numeric_features.mean())
        X[numeric_features.columns] = numeric_features
        X.drop(to_drop, axis=1, inplace=True)
        # handling categories
        to_remove = []
        for col in X.select_dtypes(include=['object']).columns:
            if (X[col].value_counts().shape[0] < 2) or (X[col].value_counts()[0] < 10):
                to_remove.append(col)
            else:
                col_to_co_percentage = X.join(y).groupby(col).co_indicator.mean()
                X[col] = X[col].map(col_to_co_percentage)
        X.drop(to_remove, axis=1, inplace=True)
        with_month = X.corrwith(y[y.co_indicator].max_mob).sort_values(key=lambda x: np.abs(x), ascending=False)
        with_co = X.corrwith(y.co_indicator).sort_values(key=lambda x: np.abs(x), ascending=False)

    top_cor = pd.concat([with_co[:300], with_month[:300]], axis=1)
    top_cor['max'] = top_cor.fillna(0).swifter.apply(lambda row: max(abs(row[0]), abs(row[1])), axis=1)

    if split:
        df = X[top_cor.index].corr().abs().stack()
    else:
        df = X_train[top_cor.index].corr().abs().stack()

    to_drop = set()
    for tup in df[(df > .7)].index:
        if (len(set(tup)) > 1) and (not set(tup) & to_drop):
            to_drop.add(top_cor.loc[list(tup), 'max'].idxmin())
    top_cor.drop(list(to_drop), axis=0, inplace=True)

    if split:
        numeric_features_train = numeric_features[top_cor.index].loc[X_train.index]
        numeric_features_test = numeric_features[top_cor.index].loc[X_test.index]
        numeric_features_train.fillna(numeric_features_train.mean(axis=0), inplace=True)
        numeric_features_test.fillna(numeric_features_train.mean(axis=0), inplace=True)
    else:
        numeric_features = numeric_features[top_cor.index]
        numeric_features.fillna(numeric_features.mean(axis=0), inplace=True)

    df = enrich_from_payments_really(payments_cashflows, payments, cut_months=label_cut)
    if split:
        X_train = pd.concat([numeric_features_train, df], join='inner', axis=1)
        X_test = pd.concat([numeric_features_test, df], join='inner', axis=1)
    else:
        X = pd.concat([numeric_features, df], axis=1)

    std_sclr = StandardScaler()
    if split:
        std_sclr.fit(X_train)
        X_train = pd.DataFrame(std_sclr.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(std_sclr.transform(X_test), index=X_test.index, columns=X_test.columns)
    else:
        std_sclr.fit(X)
        X = pd.DataFrame(std_sclr.transform(X), index=X.index, columns=X.columns)

    cols = ['ADS14V71_Audit_G231S', 'ads73a59_audit_inpl01']

    if split:
        return pd.concat([X_train.drop(columns=cols, axis=1), y], join='inner', axis=1), pd.concat(
            [X_test.drop(columns=cols, axis=1), y], join='inner', axis=1)
    else:
        return pd.concat([X.drop(columns=cols, axis=1), y], axis=1)


def main():
    features, payments = read_data_base()
    payments_features, payments_cashflows, features_copy, feature_lst = preprocess(features, payments, ret_cols=True,
                                                                                   last_seq_table=['monthly_payments'],
                                                                                   last_seq_num=6)

    payments_features_processed, payments_cashflows_processed, features_processed = final_processing(payments_features,
                                                                                                     payments_cashflows,
                                                                                                     features_copy)
    X_train, X_test = best_model_process(features_processed, payments_features_processed,
                                         payments_cashflows_processed, standardize=True)

    cph_super_enriched = CoxPHFitter()
    cph_super_enriched.fit(X_train, 'max_mob', event_col='co_indicator', step_size=0.5,
                           show_progress=True)
    alpha, beta, h_0, step_h_0 = get_h_0(X_train.iloc[:, -2:], 12, 36)
    print('predict hazard starts')
    res = predict_hazard_function(cph_super_enriched, X_test, h_0, np.arange(3, 29))
    # prediction:
    X = best_model_process(features_processed, payments_features_processed,
                           payments_cashflows_processed, standardize=True, split=False)
    print('model finished the best process')
    cph_super_enriched2 = CoxPHFitter()
    cph_super_enriched2.fit(X, 'max_mob', event_col='co_indicator', step_size=0.5,
                            show_progress=True)

    pickle.dump(cph_super_enriched, open('model.sav', 'wb'))
    alpha, beta, h_0, step_h_0 = get_h_0(X.iloc[:, -2:], 12, 36)
    res = predict_hazard_function(cph_super_enriched2, X_test, h_0, np.arange(3, 29))


if __name__ == '__main__':
    main()
