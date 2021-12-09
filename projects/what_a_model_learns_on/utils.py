import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from file_utils import parquet_to_dataframe
import numpy_financial as npf
import swifter


warnings.filterwarnings('ignore')

# ======================
# Helper Functions
# ----------------------
# Not for user usage
# ======================


def annual_to_monthly_int_rate(annual_int_rate):
    """
    Returns the monthly interest rate given the APR (Annual Percentage Rate), which is the actual metric an institution
    must disclose when discussing a loan.
    :param annual_int_rate: a float of range [0,1].
    :return: monthly int_rate: annual_int_rate / 12
    """
    return annual_int_rate / 12



def calc_monthly_pmt(df):
    """


    :param df:
    :return:
    """
    monthly_pmt = np.round(
        df.swifter.apply(lambda row: npf.pmt(annual_to_monthly_int_rate(row.int_rate * 0.01),
                                             row.term,
                                             -row.loan_amnt), axis=1))
    df.loc[:, 'monthly_pmt'] = monthly_pmt
    return df


def calc_payments(df):
    """
    Recieves df of loans and generate initial cashflow

    :param df: df of loans that contains the columns:
            ['int_rate', 'term', 'loan_amnt', 'prepaid_mob', 'co_mob', 'issue_date']
    :return: df of loans containing:
            ['issue_date', 'term', 'prepaid_mob', 'co_mob', 'interest_paid{i}', 'principal_paid{i}', 'loan_amnt', 'balance{i}']
            for i=1,...,max_term
    """
    term = int(df.term.max())

    # calculate interest_paid
    ip = df.swifter.apply(lambda row: pd.Series(
        npf.ipmt(rate=annual_to_monthly_int_rate(row.int_rate * 0.01),
                 per=range(1, int(row.term) + 1),
                 nper=int(row.term),
                 pv=-int(row.loan_amnt))), axis=1)
    ip.columns = [f'interest_paid{i + 1}' for i in range(term)]

    # calculate principal_paid
    pp = df.swifter.apply(lambda row: pd.Series(
        npf.ppmt(rate=annual_to_monthly_int_rate(row.int_rate * 0.01),
                 per=range(1, int(row.term) + 1),
                 nper=int(row.term),
                 pv=-int(row.loan_amnt))), axis=1)
    pp.columns = [f'principal_paid{i + 1}' for i in range(term)]

    # calculate balance
    balance = pp.swifter.apply(lambda row: df.loc[row.name, 'loan_amnt'] - row.cumsum(), axis=1)
    balance.columns = [f'balance{i + 1}' for i in range(term)]
    balance.insert(0, "loan_amnt", df.loan_amnt)

    return pd.concat([df.issue_date, df.term, df.prepaid_mob, df.co_mob, ip, pp, balance], axis=1)


def add_actual_pmt_columns(df):
    """
    Receives a df of initial cashflows and generate the actual payments columns.

    :param df: df of loans containing:
            ['issue_date', 'term', 'prepaid_mob', 'co_mob', 'interest_paid{i}', 'principal_paid{i}', 'loan_amnt', 'balance{i}']
            for i=1,...,max_term
    :return: the same df of loans with additional 'actual_pmt{i}' columns.
    """
    term = int(df.term.max())

    for i in range(1, term + 1):
        df[f'actual_pmt{i}'] = df[[f'principal_paid{i}', f'interest_paid{i}']].sum(axis=1, skipna=True)
    return df


def edit_charged_off_loans_payments(df):
    """
    Receives a df of cashflows and edit the 'actual_pmt' columns according to 'co_mob',
    in a way that if a loan charges off in mob x, the actual_pmt of this loan will be 0 from mob x and forward.

    :param df: df of loans containing payments information.
    :return: edited df.
    """
    term = int(df.term.max())

    df.co_mob = df.co_mob.fillna(-1)
    df = df.swifter.apply(lambda x: pd.Series(np.hstack((x.loc[:f'actual_pmt{int(x.co_mob)}'][:-1],
                                               np.zeros(term + 1 - int(x.co_mob)))),
                                              index=x.index) if x.co_mob != -1 else x,
                          axis=1)
    df.loc[df.co_mob == -1, 'co_mob'] = np.nan
    return df


def edit_prepaid_loans_payments(df):
    """
    Receives a df of cashflows and edit the 'actual_pmt' columns according to 'prepaid_mob',
    in a way that if a loan prepais in mob x, the actual_pmt of this loan in mob x will be actual_pmt{x} + balanc{x},
    and in all the following mobs, it's actual_pmt will be 0.

    :param df: df of loans containing payments information.
    :return: edited df.
    """
    term = int(df.term.max())

    df.prepaid_mob = df.prepaid_mob.fillna(-1)
    df = df.swifter.apply(lambda x: pd.Series(np.hstack((x.loc[:f'actual_pmt{int(x.prepaid_mob)}'][:-1],
                                                         (x[f'actual_pmt{int(x.prepaid_mob)}'] + x[
                                                             f'balance{int(x.prepaid_mob)}']),
                                                         np.zeros(term + 1 - int(x.prepaid_mob) - 1))),
                                              index=df.columns)
    if x.prepaid_mob != -1 else x, axis=1)
    df.loc[df.prepaid_mob == -1, 'prepaid_mob'] = np.nan
    return df


def calc_months_diff(date_series, year, month):
    years = pd.DatetimeIndex(date_series).year
    months = pd.DatetimeIndex(date_series).month
    return 12 * (year - years) + (month - months)


def calc_prepaid_mob(orig_prepaid_mob, info_mob, term, co_mob):
    if not np.isnan(co_mob):
        if co_mob < info_mob:
            return orig_prepaid_mob

    if np.isnan(orig_prepaid_mob):
        if info_mob < term:
            return info_mob
    else:
        if info_mob < orig_prepaid_mob:
            return info_mob

    return orig_prepaid_mob


def set_information_date(df, info_year, info_month):
    info_date_mobs = calc_months_diff(df.issue_date, info_year, info_month)
    df['info_date_mob'] = info_date_mobs
    assert (df['info_date_mob'] > 0).all()
    prepaid_mobs = pd.concat([df.prepaid_mob, df.info_date_mob, df.term, df.co_mob], axis=1).swifter.apply(lambda x: calc_prepaid_mob(x.prepaid_mob, x.info_date_mob, x.term, x.co_mob), axis=1)
    df.drop(columns=['info_date_mob'], inplace=True)
    df['prepaid_mob'] = prepaid_mobs
    return edit_prepaid_loans_payments(df)

def calc_similarity(x, regular, initial):
    return ((x - initial).sum()) ** 2 / (len(x) + regular)


# ======================
# Main Functions
# ======================


def generate_cashflows(df, const_int_rate=0):
    """
    Given a df of loans, calculate its cashflows.

    :param df:
    :return:
    """
    df = df.copy()
    if const_int_rate:
        df['orig_int_rate'] = df.int_rate
        df.loc[:, 'int_rate'] = const_int_rate
    return edit_charged_off_loans_payments(
                add_actual_pmt_columns(
                    calc_payments(
                        calc_monthly_pmt(df))))


def calc_irr(cashflows, info_date=None):
    """
    Given cashflows of loans calculate the loans' annual IRR.

    :param cashflows:
    :return:
    """
    cashflows = cashflows.copy()

    if info_date:
        cashflows = set_information_date(cashflows, info_date[0], info_date[1])
    else:
        cashflows = edit_prepaid_loans_payments(cashflows)

    cashflows_payments = cashflows.filter(like='actual_pmt')
    cashflows_payments.insert(0, 'loan_amnt', -cashflows.loan_amnt)

    return cashflows_payments.swifter.apply(npf.irr, axis=1).swifter.apply(lambda irr: ((irr + 1) ** 12 - 1) * 100,
                                                                           axis=1).fillna(-100)
def calc_should(feature, y, regular=0, initial=0.5):
    """
    The function gets a certain binary feature column from a dataset and the label vector,
    and returns the "should" score of that feature, which is the gain of an XGBoost
    tree node that splits according to the feature.

    regular is an optional parmeter that allows for adding regularization to the calculation
    of the similarity score.
    """

    assert (np.sort(feature.unique()) == [0, 1]).all(), "feature should only have 0s and 1s"
    assert len(feature) == len(y), "The dataframe and the label vector are not the same length"


    zeros = y[feature == 0]
    ones = y[feature == 1]

    zeros_similarity = calc_similarity(zeros, regular, initial)
    ones_similarity = calc_similarity(ones, regular, initial)
    root_similarity = calc_similarity(y, regular, initial)

    gain = zeros_similarity + ones_similarity - root_similarity

    return gain