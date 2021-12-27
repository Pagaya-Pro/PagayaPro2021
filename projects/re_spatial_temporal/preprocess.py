import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import pickle
import sys


def split_train_val_test(df, train_start_date, validation_start_date, test_start_date, test_end_date, date_col_name,
                         shuffle=False, shuffle_rate=0.5):
    print('train-val-test split')
    interval1 = df[(df[date_col_name] >= train_start_date) & (df[date_col_name] < validation_start_date)]
    interval2 = df[(df[date_col_name] >= validation_start_date) & (df[date_col_name] < test_start_date)]
    interval3 = df[(df[date_col_name] >= test_start_date) & (df[date_col_name] < test_end_date)]

    if shuffle:
        interval2, interval3 = train_test_split(interval2, test_size=shuffle_rate, random_state=42)

    return interval1, interval2, interval3


def drop_rows_deed(df_deed):
    print('dropping rows from deed')
    tot = len(df_deed)

    df_deed = df_deed[(df_deed.SaleAmt >= 30000) &
                      (df_deed.SaleAmt <= 600000)]

    df_deed = df_deed[~df_deed[['PropertyID', 'SaleAmt', 'SaleDate']].duplicated()]
    df_deed = df_deed[~df_deed[['PropertyID', 'SaleAmt', 'RecordingDate']].duplicated()]

    print(f'{tot - len(df_deed)} rows deleted.')
    return df_deed


def drop_rows_assessor(df_assessor):
    print('dropping rows from assessor')
    tot = len(df_assessor)

    df_assessor = df_assessor.drop(df_assessor.index[(df_assessor.BathFull > 6) |
                                                     (df_assessor.Bedrooms < 3) |
                                                     (df_assessor.Bedrooms > 6) |
                                                     (df_assessor.TotalRooms > 12) |
                                                     (df_assessor.GarageParkingNbr > 4) |
                                                     (df_assessor.SumResidentialUnits > 2) &
                                                     (~df_assessor.SitusGeoStatusCode.isin(['A', 'B'])) |
                                                     (~df_assessor.LandUseCode.isin([1000, 1001, 1002, 1999])) |
                                                     (df_assessor.MobileHomeInd == 'T') |
                                                     (df_assessor.StoriesNbrCode > 400)] |
                                   (df_assessor.AirConditioningCode.isin([3, 6, 10])), axis=0)

    print(f'{tot - len(df_assessor)} rows deleted.')

    return df_assessor


def create_combined_dataset(df_deed, df_assessor):
    # Row Filtering
    df_deed = drop_rows_deed(df_deed)
    df_assessor = drop_rows_assessor(df_assessor)

    df_deed = df_deed[['PropertyID', 'SaleDate', 'SaleAmt']]

    # add features from assessor to deed
    # note, almost every PropertyID will appear twice due to AssdYear 2020 and 2021
    combined = df_deed.join(df_assessor.set_index('PropertyID', drop=True), on='PropertyID', how='inner')

    print(f'There are {len(combined)} rows in the combined dataset')
    return combined


def create_dataset(start_dates, end_train_dates, end_test_dates, end, deed_path='deed_6037.parquet',
                   assessor_path='assessor_6037.parquet'):
    # load datasets
    deed = pd.read_parquet(deed_path)
    assessor = pd.read_parquet(assessor_path)

    # building dataset
    combined = create_combined_dataset(deed, assessor)

    for i in range(len(start_dates)):
        # split based on dates. Note, duplicates for AssdYear are still here
        train, val, test = split_train_val_test(combined.copy(), start_dates[i], end_train_dates[i], end_test_dates[i],
                                                end,
                                                'SaleDate', shuffle=True)

        # Take the right assessment year
        train['SaleYear'] = train.SaleDate.apply(lambda x: x.year)
        val['SaleYear'] = val.SaleDate.apply(lambda x: x.year)
        test['SaleYear'] = test.SaleDate.apply(lambda x: x.year)

        train['keep2020'] = train.SaleYear <= 2020
        val['keep2020'] = val.SaleYear <= 2020
        test['keep2020'] = test.SaleYear <= 2020

        train = train[((train.AssdYear == 2020) & train.keep2020) | ((train.AssdYear == 2021) & (~train.keep2020))] \
            .set_index('PropertyID').sort_index()
        val = val[((val.AssdYear == 2020) & val.keep2020) | ((val.AssdYear == 2021) & (~val.keep2020))] \
            .set_index('PropertyID').sort_index()
        test = test[((test.AssdYear == 2020) & test.keep2020) | ((test.AssdYear == 2021) & (~test.keep2020))] \
            .set_index('PropertyID').sort_index()

        train = train.drop(['SaleYear', 'keep2020'], axis=1)
        val = val.drop(['SaleYear', 'keep2020'], axis=1)
        test = test.drop(['SaleYear', 'keep2020'], axis=1)

        # Don't train on PropertyIDs with 8 or more transactions. On the last check
        train = train.drop(train.index.value_counts()[train.index.value_counts() >= 100].index)

        #         with open(f'split_on_{validation_start_date.strftime("%Y_%m_%d")}_short_train.pkl', 'wb') as f:

        with open(f'split_on_{end_train_dates[i].strftime("%Y_%m_%d")}_short_train.pkl', 'wb') as f:
            print('create', end_train_dates[i].strftime("%Y_%m_%d"))
            pickle.dump([train, val, test], f)


def calc_index(deed, calc_from=None, calc_to=None, return_index_from=None, return_index_to=None,
               propertyID_is_index=True, verbose=False):
    """
    Given a list of transactions deed, calculates the price index using BMN method.
    INPUT:
     - deed (DF):
             Dataframe of property sales - should contain property ID as index, sale price and sale date (as dt). If property ID is no the index,
             use propertyID_is_index = True.
     - calc_from,calc_to (date times):
             Date range in which to calculate the index. If Nones, taken to be the dates of the first/last sales in deed.
     - return_index_from, return_index_to (date times):
             Date range for the output index Series. Values will be normalized to last date. If Nones, values are taken to be calc_from/calc_to
     - PropertyID_is_index (boolean):
             If False, reindexes deed so propertyID is the index, for faster performance.
     - verbose (boolean):
             If True, prints progress along unique properties.

    OUTPUT:
     - month_indices (Series):
             The price index, with the date as series index.

    Assumpsions: all months are covered between calc_from to calc_to.
    TODO: create BMN columns only for unique dates apearing in the data, and not the whole range.
    """

    if calc_from is None:
        calc_from = deed.SaleDate.min()

    if calc_to is None:
        calc_to = deed.SaleDate.max()

    if return_index_from is None:
        return_index_from = calc_from

    if return_index_to is None:
        return_index_to = calc_to

    # round to start of months
    if calc_from.day > 1:
        calc_from = calc_from - pd.offsets.MonthBegin(1)
    if calc_to.day > 1:
        calc_to = calc_to - pd.offsets.MonthBegin(1)
    if return_index_from.day > 1:
        calc_from = calc_from - pd.offsets.MonthBegin(1)
    if return_index_to.day > 1:
        calc_to = calc_to - pd.offsets.MonthBegin(1)

    deed_copy = deed[(deed.SaleDate >= calc_from) & (deed.SaleDate <= calc_to)].copy()  # leave deed unharmed

    # Index by PropertyID for better performance
    if not propertyID_is_index:
        deed_copy = deed_copy.set_index('PropertyID')

    find_duplicates = deed_copy.reset_index(level=0)[['PropertyID', 'SaleAmt', 'SaleDate']].duplicated()
    deed_copy = deed_copy[['SaleAmt', 'SaleDate']][~find_duplicates.values]

    deed_copy['period'] = (deed_copy.SaleDate.dt.to_period('M') - calc_from.to_period('M')).map(
        lambda x: x.n)  # calc period (months since start)
    last_period = (calc_to.to_period('M') - calc_from.to_period('M')).n + 1
    t = []
    t_tag = []
    ratios = []
    uniqe_properties = deed_copy.index.unique()
    for c, house_id in enumerate(uniqe_properties):
        if verbose and c % 10000 == 0:
            print(f'{c} out of {len(uniqe_properties)}')
        transactions = deed_copy.loc[house_id:house_id].sort_values(by='period')
        for i in range(len(transactions) - 1):
            first_sale_period = transactions.iloc[i].period
            second_sale_priod = transactions.iloc[i + 1].period
            if first_sale_period != second_sale_priod:
                t.append(first_sale_period)
                t_tag.append(second_sale_priod)
                ratios.append(transactions.iloc[i + 1].SaleAmt / transactions.iloc[i].SaleAmt)
        n = len(t)
        X = np.zeros((n, last_period))
        X[range(n), t] = -1
        X[range(n), t_tag] = 1
        r = np.log(np.array(ratios))

    LR = LinearRegression(fit_intercept=False).fit(X_train, r_train)
    index_indice_start = (return_index_from.to_period('M') - calc_from.to_period('M')).map(lambda x: x.n)
    drange = pd.date_range(return_index_from, return_index_to, freq='MS')
    keep_ind = range(index_indice_start, index_indice_start + len(drange))
    b = LR.coef_[keep_ind]
    #     drange = drange_full[keep_ind]
    b = b - b[-1]
    B = np.exp(b)
    month_indices = pd.Series(data=B, index=drange)
    return month_indices


start_dates = pd.date_range(pd.to_datetime('20160101', format='%Y%m%d'), pd.to_datetime('20170701', format='%Y%m%d'),
                            freq='MS')
end_train_dates = pd.date_range(pd.to_datetime('20200101', format='%Y%m%d'),
                                pd.to_datetime('20210701', format='%Y%m%d'), freq='MS')
end_test_dates = pd.date_range(pd.to_datetime('20200401', format='%Y%m%d'), pd.to_datetime('20211001', format='%Y%m%d'),
                               freq='MS')

np.set_printoptions(threshold=sys.maxsize)
create_dataset(start_dates, end_train_dates, end_test_dates, datetime(2023, 6, 1))


