def add_actual_pmt_columns(df):
    """
    Receives a df that is the output of create_initial_cashflows_template, and adds columns of actual_pmt{i} for each term.
    The actual_pmt columns would be defined as:
    actual_pmt{i} = principal_paid{i} + interest_paid{i}

    Returns: df with additional columns
    """
    for i in range(1, 61):
        df[f'actual_pmt{i}'] = df[[f'principal_paid{i}', f'interest_paid{i}']].sum(axis=1, skipna=True)
    return df


def edit_charged_off_loans_payments(df):
    """
    Receives a df that is the output of add_actual_pmt_columns, and edits the actual_pmt columns,
    in a way that if a loan charges off in mob x, the actual_pmt of this loan will be 0 from mob x and forward.

    Returns: edited df
    """
    df.co_mob = df.co_mob.fillna(-1)
    df = df.swifter.apply(lambda x: np.hstack((x.loc[:f'actual_pmt{int(x.co_mob)}'][:-1],
                                               np.zeros(61 - int(x.co_mob)))) if x.co_mob != -1 else x,
                          axis=1)
    df.loc[df.co_mob == -1, 'co_mob'] = np.nan
    return df


def edit_prepaid_loans_payments(df):
    """
    Receives a df that is the output of edit_charged_off_loans_payments, and edits the actual_pmt columns,
    in a way that if a loan prepais in mob x, the actual_pmt of this loan in mob x will be actual_pmt{x} + balanc{x},
    and in all the following mobs, it's actual_pmt will be 0

    Returns: edited df
    """
    df.prepaid_mob = df.prepaid_mob.fillna(-1)
    df = df.swifter.apply(lambda x: pd.Series(np.hstack((x.loc[:f'actual_pmt{int(x.prepaid_mob)}'][:-1],
                                                         (x[f'actual_pmt{int(x.prepaid_mob)}'] + x[
                                                             f'balance{int(x.prepaid_mob)}']),
                                                         np.zeros(61 - int(x.prepaid_mob) - 1))), index=df.columns)
    if x.prepaid_mob != -1 else x, axis=1)
    df.loc[df.prepaid_mob == -1, 'prepaid_mob'] = np.nan
    return df

