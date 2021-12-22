import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from file_utils import parquet_to_dataframe
import numpy_financial as npf
import swifter
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score as bas
import shap
from scipy.stats import gmean
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp, ttest_ind
from kneed import KneeLocator
import copy
from sklearn import metrics

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

def double_r_leaf(leaf_flags, len_data):
    return (len(leaf_flags) * abs(leaf_flags.mean()-0.5)) / len_data

def double_r_tree(leaves, flag):
    leaves_score = []
    for leaf in range(1, max(leaves) + 1):
        indx = (np.array(leaves) == leaf)
        leaf_flags = flag[indx]
        if len(leaf_flags) > 0:
            leaves_score.append(double_r_leaf(leaf_flags, len(flag)))
    return np.array(leaves_score).sum()

def double_r_model(leaves, X, flag):
    """
    :param loans_leaves:
        list of lists--for every sample, for every tree, the leaf of the sample
    :param X:
        models' features
    :return:
    """
    if len(leaves.shape) == 1:
        leaves = leaves.reshape(-1,1)
    trees_leaves = leaves.T
    trees_score = []
    for tree in tqdm(trees_leaves):
        trees_score.append(double_r_tree(tree, flag))
    return trees_score


def equalize(small_idx, large_idx, seed=42):
    n_small = len(small_idx)

    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    large_idx_to_keep = rs.choice(large_idx, n_small, replace=False)
    idx_to_keep = np.concatenate((small_idx, large_idx_to_keep))

    return idx_to_keep

def similarity(data, regular, initial):

    return ((data - initial).sum()) ** 2 / (len(data) + regular)

def gain(X, y, flag, initial=0, regular=0):

    zeros = y[flag == 0]
    ones = y[flag == 1]

    zeros_similarity = similarity(zeros, regular, initial)
    ones_similarity = similarity(ones, regular, initial)
    root_similarity = similarity(y, regular, initial)

    return zeros_similarity + ones_similarity - root_similarity


def get_dependent_features(X, flag, acc_thld=0.75, dec_thld=0.8):
    dependent_features = []
    first_balances_acc = None

    # Predict flag by using the remaining features with XGBoost classification
    while len(dependent_features) < X.shape[1]:
        X_cur = X.drop(columns=dependent_features)
        flag_clf = xgb.XGBClassifier(n_estimators=7, random_state=123)
        flag_clf.fit(X_cur, flag)
        flag_preds = flag_clf.predict(X_cur)

        # Drop features if balanced accuracy passes threshold
        balanced_acc = bas(flag, flag_preds, adjusted=False)

        if first_balances_acc is not None and (balanced_acc / first_balances_acc) < dec_thld:
            break

        # Calculate SHAP values
        explainer = shap.Explainer(flag_clf)
        shap_values = explainer(X_cur)
        df_shap_values = pd.DataFrame(shap_values.values, columns=X_cur.columns, index=X_cur.index).abs()
        df_shap_values_norm = df_shap_values.divide(df_shap_values.sum(axis=1), axis=0)
        shap_values_norm = df_shap_values_norm.mean(axis=0)
        max_shap_feature = X_cur.columns[shap_values_norm.argmax()]
        if first_balances_acc is None:
            first_balances_acc = balanced_acc
            if balanced_acc < acc_thld:
                break
        dependent_features.append(max_shap_feature)

    return dependent_features


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

def should(X, y, flag, regular=0, verbose=False):
    """
    This function calculates the "should" score we came up with. Given a flag feature and the target vector,
    the "should" score intends to measure how useful it is to use the flag as a feature when training a
    model to predict the target.
    :param y:
        The target vector.
    :param flag:
        A column the same length as y with 0s and 1s.
    :param regular:
        This is a regularization parameter.
        It should have the same value as the regularization parameter of the actual model that will be used to predict y.
        The default value is 0.
    :return:
        "should" score
    """
    normalize = []

    for i, initial in enumerate([np.min(y) - 1, np.max(y) + 1]):
        sim = similarity(y, regular, initial)
        normalize.append(sim)
        txt = 'min'
        if i==1:
            txt = 'max'
        if verbose:
            print(f'Similarity when initial is {txt}: {sim}')

    g = gain(X, y, flag)
    if verbose:
        print(f'Gain: {g}')
    return g / gmean(normalize)


def can_simplicity(X, y, flag, verbose=False, plot_tree=False, max_max_depth=6, seed=42, test_size=0.33):
    """
    This function calculates the "can*simplicity" score we came up with. Given a flag feature and the training dataset,
    the "can*simplicity" score intends to measure how easy it is to predict the flag from the data.
    The function trains a single XGBoost tree for each depth from 1 to max_max_depth, train to predict to flag
    from the data. For each model, it calculates an accuracy score and normalizes it by the tree's depth.
    The final score is the max score obtained after normalizing by depth.
    :param X:
        The training data.
    :param y:
        The model's labels (not really necessary; used for standardization)
    :param flag:
        A column the same length as the data with 0s and 1s.
    :param verbose:
        If True, the function prints out the results for each model it trains.
    :param plot_trees:
        If True, the functions prints out the decision tree for each model.
    :param max_max_depth:
        The upper bound for the max_depth of the models the function builds.
    :param seed:
        Random state seed.
    :param test_size:
        The test size when splitting the data.
    :return:
        The function returns a dict with the following keys:
            Score: The score of the winning model.
            Accuracy: The accuracy of the winning model.
            Depth: The depth of the winning model.
    """
    can_list = []
    products = []
    models = []

    zeros = X[flag == 0].index
    ones = X[flag == 1].index
    if len(zeros) != len(ones):
        if len(zeros) < len(ones):
            idx_to_keep = equalize(zeros, ones, seed)
        else:
            idx_to_keep = equalize(ones, zeros, seed)
        idx_to_keep_bool = X.index.isin(idx_to_keep) == True
        X = X[idx_to_keep_bool]
        y = y[idx_to_keep_bool]
        flag = flag[idx_to_keep_bool]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        flag,
        test_size=test_size,
        random_state=seed)

    for dep in range(1, max_max_depth + 1):

        model = xgb.XGBRegressor(
            eval_metric='logloss',
            objective="binary:logistic",
            random_state=seed,
            n_estimators=1,
            max_depth=dep)

        model.fit(X_train, y_train)
        models.append(model)
        probs = model.predict(X_test)
        preds = np.zeros(len(probs))
        preds[probs > 0.5] = 1

        acc = accuracy_score(y_test, preds)
        simple = 1 / np.sqrt(dep)
        score = 2 * abs(acc - 0.5)
        product = score * simple
        can_list.append(score)
        products.append(product)

        if verbose:
            print(f'Depth: {dep}')
            print(f'\tAccuracy: {acc}')
            print(f'\tNormed accuracy: {score}')
            print(f'\tSimplicity: {simple}')
            print(f'\tScore*Simplicity: {product}')

    max_prod_id = np.argmax(products)
    max_prod = products[max_prod_id]
    max_score_id = np.argmax(can_list)
    max_score = can_list[max_score_id]

    if plot_tree:
        fig, ax = plt.subplots(figsize=(8+4*max_prod_id, 8+4*max_prod_id))
        xgb.plot_tree(models[max_prod_id], ax=ax)
        plt.title(f'Can*Simplicity best tree (depth= {max_prod_id+1}, accuracy={can_list[max_prod_id]})')
        plt.show();

    if verbose:
        print(
            f'Best can*simplicity score: {max_prod:.4f} for depth: {max_prod_id + 1} and can score {can_list[max_prod_id]:.4f}')
        print(f'Best can score: {max_score:.4f} for depth: {max_score_id + 1}')

    res = {
        'score': max_prod,
        'accuracy': can_list[max_prod_id],
        'depth': max_prod_id+1
    }
    return res

def should_can_simple(X, y, flag, regular=0, max_max_depth=6, seed=42, test_size=0.33):
    s = should(X, y, flag, regular)
    c = can_simplicity(X, y, flag, max_max_depth=max_max_depth, seed=seed, test_size=test_size)['score']
    return s*c

def compare_preds(X, y, flag, model=None, alpha=0.01):
    if not model:
        model = xgb.XGBRegressor(
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth)
        model.fit(X, y)

    preds = model.predict(X)

    preds_0 = preds[flag == 0]
    preds_1 = preds[flag == 1]

    res = dict()

    res['diff'] = np.mean(preds_1) - np.mean(preds_0)
    res['ttest'] = (ttest_ind(preds_0, preds_1)[1] < alpha) * 1
    res['ks'] = (ks_2samp(preds_0, preds_1)[1] < alpha) * 1

    return res



def double_r(X, y, flag, model=None, seed=42, n_estimators=20, max_depth=6, plot_acc=False):
    """
    :param X: Model features
    :param y: Model labels
    :param flag: Flag to be tested
    :param seed: Random seed (default=42)
    :return: Double R score
    """
    if not model:
        if len(X) < 50000:
            print("Sample should be larger than 50,000")


        if len(X) < 50000:
            print("Sample is too imbalanced. Either increase sample or reduce imbalance in flag.")

        model = xgb.XGBRegressor(
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth)
        model.fit(X, y)

    zeros = X[flag == 0].index
    ones = X[flag == 1].index
    if len(zeros) != len(ones):
        if len(zeros) < len(ones):
            idx_to_keep = equalize(zeros, ones, seed)
        else:
            idx_to_keep = equalize(ones, zeros, seed)
        idx_to_keep_bool = X.index.isin(idx_to_keep) == True
        X = X[idx_to_keep_bool]
        y = y[idx_to_keep_bool]
        flag = flag[idx_to_keep_bool]

    leaves = model.apply(X)
    res = double_r_model(leaves, X, flag)

    if plot_acc:
        acc_df = pd.DataFrame(index=np.arange(1, n_estimators+1), data=res)
        sns.scatterplot(data=acc_df)
        plt.xlabel('Tree number')
        plt.ylabel('Tree score')
        plt.title('R2C tree score for each tree in the model')
        plt.show();

    return np.mean(res)


def SHAP_score(X, y, flag, acc_thld=0.75, dec_thld=0.8, print_dependent=False):
    # Find dependent features
    dependent_features = get_dependent_features(X, flag, acc_thld=acc_thld, dec_thld=dec_thld)
    if print_dependent:
        print(f'dependent features: {dependent_features}')
    # Calculate Should
    X_flag = X.drop(dependent_features, axis=1)
    X_flag['flag'] = flag
    flag_model = xgb.XGBRegressor(n_estimators=20, random_state=42)
    flag_model.fit(X_flag, y)

    flag_explainer = shap.Explainer(flag_model)
    flag_shap_values = flag_explainer(X_flag)

    df_flag_shap_values = pd.DataFrame(flag_shap_values.values, columns=X_flag.columns, index=X_flag.index)
    should = (df_flag_shap_values.flag.abs() / df_flag_shap_values.abs().sum(axis=1)).mean()

    # Calculate Can
    can_model = xgb.XGBClassifier(n_estimators=7, random_state=111)
    can_model.fit(X, flag)
    proba = can_model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(flag, proba, drop_intermediate=False)
    can = bas(flag, (proba > thresholds[np.argmax(tpr - fpr)]).astype(int), adjusted=False)


    # Calculate Difficulty
    most_important = copy.deepcopy(dependent_features)
    print(f'{len(most_important)} dependent features: {most_important}')
    accs = []
    for i in range(len(dependent_features)):
        dependent_model = xgb.XGBClassifier(n_estimators=20, random_state=42)
        dependent_model.fit(X[most_important], flag)
        dependent_preds = dependent_model.predict(X[most_important])
        accuracy = bas(flag, dependent_preds, adjusted=False)
        accs.append(accuracy)

        explainer = shap.Explainer(dependent_model)
        shap_values = explainer(X[most_important])
        df_shap_values = pd.DataFrame(shap_values.values, columns=most_important,
                                      index=X[most_important].index).abs().mean(axis=0)
        drop_feature = df_shap_values.idxmin()
        most_important.remove(drop_feature)

    if len(accs) == 0:
        can_explainer = shap.Explainer(can_model)
        can_shap_values = pd.DataFrame(can_explainer(X).values, columns=X.columns,
                                      index=X.index).abs()
        can_agg_shap = can_shap_values.divide(can_shap_values.sum(axis=1), axis='rows').mean(axis='rows').sort_values(ascending=False).to_numpy()
        difficulty = KneeLocator(range(1, len(can_agg_shap) + 1), can_agg_shap, curve='convex',
                                 direction='decreasing').knee
        if difficulty == 1 or difficulty == len(can_agg_shap):
            difficulty = KneeLocator(1, range(len(can_agg_shap) + 1), can_agg_shap, curve='concave',
                                     direction='increasing').knee

    elif len(accs) == 1:
        difficulty = 1
    elif len(accs) == 2:
        if accs[1] / accs[0] >= 0.85:
            difficulty = 1
        else:
            difficulty = 2
    else:
        difficulty = KneeLocator(range(1, len(accs) + 1), accs[::-1], curve='concave', direction='increasing').knee
        if difficulty == 1 or difficulty == len(accs):
            difficulty = KneeLocator(1, range(len(accs) + 1), accs[::-1], curve='convex', direction='increasing').knee
        elif difficulty is None:  # All accuracies are equal
            difficulty = 1

    # Return results
    return should, can, difficulty


def can_difficulty(X, flag):
    """
    TEMPORARY for rerunning flags
    :param X:
    :param y:
    :param flag:
    :param calc_can:
    :return:
    """
    # Calculate Can
    can_model = xgb.XGBClassifier(n_estimators=7, random_state=111)
    can_model.fit(X, flag)
    proba = can_model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(flag, proba, drop_intermediate=False)
    can = bas(flag, (proba > thresholds[np.argmax(tpr - fpr)]).astype(int), adjusted=False)

    # Calculate Difficulty of 0 dependent features.
    can_explainer = shap.Explainer(can_model)
    can_shap_values = pd.DataFrame(can_explainer(X).values, columns=X.columns,
                                  index=X.index).abs()
    can_agg_shap = can_shap_values.divide(can_shap_values.sum(axis=1), axis='rows').mean(axis='rows').sort_values(ascending=False).to_numpy()
    difficulty = KneeLocator(range(1, len(can_agg_shap) + 1), can_agg_shap, curve='convex', direction='decreasing').knee
    if difficulty == 1 or difficulty == len(can_agg_shap):
        difficulty = KneeLocator(1, range(len(can_agg_shap) + 1), can_agg_shap, curve='concave',
                                 direction='increasing').knee

    # Return results
    return can, difficulty