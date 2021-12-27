import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import psutil
import socket
import tempfile
import functools
import math
import swifter
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from file_utils import parquet_to_dataframe

def survivals_to_cashflows(survivals):
    """
    Receives a df that with the target values needed to create cashflows pivot table.
    Returns: cashflows table
    """

    initial_template = create_initial_cashflows_template(survivals) #- we created the template already
    template_with_actual_pmt = add_actual_pmt_columns(initial_template)
    template_with_actual_pmt_and_co_adjustment = edit_charged_off_loans_payments(template_with_actual_pmt)
    adjusted_template = edit_prepaid_loans_payments(template_with_actual_pmt_and_co_adjustment)
    pmt_cols = [col for col in adjusted_template if 'actual_pmt' in col]
    cashflows  = pd.concat([-survivals.reset_index().loan_amnt, adjusted_template[pmt_cols]], axis=1)
    return cashflows.fillna(0)


def add_actual_pmt_columns(df):
    """
    Receives a df that is the output of create_initial_cashflows_template, and adds columns of actual_pmt{i} for each term.
    The actual_pmt columns would be defined as:
    actual_pmt{i} = principal_paid{i} + interest_paid{i}


    Returns: df with additional columns
    """
    df_copy = df.copy()
    print('started actual payments')
    for i in range(1, 61):
        int_rate_col = f'interest_paid{i}'
        principal_col = f'principal_paid{i}'
        monthly_pmt_col = f'actual_pmt{i}'
        df_copy[monthly_pmt_col] = df_copy[int_rate_col] + df_copy[principal_col]
        df_copy[monthly_pmt_col] = df_copy[monthly_pmt_col].fillna(0)

    return df_copy


def edit_charged_off_loans_payments(df):
    """
    Receives a df that is the output of add_actual_pmt_columns, and edits the actual_pmt columns,
    in a way that if a loan charges off in mob x, the actual_pmt of this loan will be 0 from mob x and forward.


    Returns: edited df
    """
    df_copy = df.copy()
    print('started CO')
    for i in range(1, 61):
        monthly_pmt_col = f'actual_pmt{i}'
        df_copy.loc[((df_copy['co_mob'].notna()) & (df_copy['co_mob'] <= i)), monthly_pmt_col] = 0

    return df_copy


def edit_prepaid_loans_payments(df):
    """
    Receives a df that is the output of edit_charged_off_loans_payments, and edits the actual_pmt columns,
    in a way that if a loan prepais in mob x, the actual_pmt of this loan in mob x will be actual_pmt{x} + balance{x},
    and in all the following mobs, it's actual_pmt will be 0

    Returns: edited df
    """
    df_copy = df.copy()
    print('started prepaid')
    for i in range(1, 61):
        monthly_pmt_col = f'actual_pmt{i}'
        mask_forward_months = ((df_copy['prepaid_mob'].notna()) & (df_copy['prepaid_mob'] < i))
        mask_current_month = ((df_copy['prepaid_mob'].notna()) & (df_copy['prepaid_mob'] == i))
        df_copy.loc[mask_current_month, monthly_pmt_col] = df_copy.loc[mask_current_month][f'balance{i}'] + \
                                                           df_copy.loc[mask_current_month][monthly_pmt_col]
        df_copy.loc[mask_forward_months, monthly_pmt_col] = 0

    return df_copy


def calc_payments(df):
    ip = df.swifter.apply(lambda row: npf.ipmt(rate=row.int_rate / 1200, per=range(1, int(row.term)+2), nper=int(row.term), pv=-int(row.loan_amnt)), axis=1)
    pp = df.swifter.apply(lambda row: npf.ppmt(rate=row.int_rate / 1200, per=range(1, int(row.term)+2), nper=int(row.term), pv=-int(row.loan_amnt)), axis=1)
    balance = df.swifter.apply(lambda row: npf.fv(rate=row.int_rate / 1200, pmt=row.monthly_pmt, nper=range(int(row.term) + 1), pv=-int(row.loan_amnt)), axis=1)
    return ip, pp, balance

def create_initial_cashflows_template(survivals):
    ip, pp, balance = calc_payments(survivals)
    payments = pd.concat([survivals.account_id, survivals.prepaid_mob, survivals.co_mob,ip,pp,balance], axis=1).rename(columns={0: 'interest_paid', 1: 'principal_paid', 2:'balance'})
    df1 = pd.DataFrame(payments['interest_paid'].to_list(), columns=[f'interest_paid{i}' for i in range(max(len(x) for x in payments['interest_paid'].to_list()))]) #, index=payments.account_id)
    df2 = pd.DataFrame(payments['principal_paid'].to_list(), columns=[f'principal_paid{i}' for i in range(max(len(x) for x in payments['principal_paid'].to_list()))]) #, index=payments.account_id)
    df3 = pd.DataFrame(payments['balance'].to_list(), columns=[f'balance{i}' for i in range(max(len(x) for x in payments['balance'].to_list()))]) #, index=payments.account_id)
    pp_mob = pd.DataFrame(payments['prepaid_mob']) # , index=payments.account_id)
    co_mob = pd.DataFrame(payments['co_mob'])
    df4 = pd.concat([df1, df2, df3, pp_mob.reset_index(), co_mob.reset_index()], axis=1)
    return df4