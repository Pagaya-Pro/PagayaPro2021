import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

BEST_THRESHOLD = 0.45


class CoClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier for loans' Charge-off prediction using Logistic Regression
    The classifier can be used for inference only for 'new' data, meaning that the earliest date for prediction must be
    later than the latest data in the training data.
    """

    def __init__(self):
        """
        Initializes a model with a scaler and logistic regression model
        """
        self._model = make_pipeline(StandardScaler(), LogisticRegression())
        self._max_date = None
        self._max_encoded_date = None
        self._occupation_replace = None

    def _preprocess(self, X: pd.DataFrame, train: bool = True) -> pd.DataFrame:
        """
        Preprocesses the data before fitting or predicting from it
        :param X: Pandas DataFrame of loans data. If train == True contains a binary column of 'CO'
        :param train: boolean, True iff called from fit
        :return: Pandas preprocessed DataFrame
        """
        cols_to_drop = ['all803', 'all807', 'all801', 'inquiries_last6_months', 'total_inquiries', 'g099s', 'hr12s',
                        'ale720', 'iln724', 'ale724', 'monthly_pmt', 'rev302', 'bac302', 'hr09s', 'rep071', 'rep901',
                        's004s', 'borrower_city']
        X = X.drop(columns=cols_to_drop)

        X = X.fillna({'occupation': 'no_occupation'})
        X['issue_date'] = LabelEncoder().fit_transform(X['issue_date'])

        if train:
            self._max_encoded_date = X['issue_date'].max()

            self._occupation_replace = X.groupby('occupation')['CO'].mean()
            self._occupation_replace.loc[r'.*'] = X['CO'].mean()

        else:
            X['issue_date'] = X['issue_date'] + self._max_encoded_date

        X['occupation'] = X['occupation'].replace(self._occupation_replace, regex=True)
        X['loan_amnt'] = np.log(X['loan_amnt'].values)

        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fits a Logistic regression model for charge-off prediction
        :param X: Pandas DataFrame of loans data, containing the following columns: all804 - float64
                                                                                    rep501 - float64
                                                                                    hr24s - float64
                                                                                    rt24s - float64
                                                                                    iln703 - float64
                                                                                    hr06s - float64
                                                                                    g099a - float64
                                                                                    iln720 - float64
                                                                                    percent_funded - float64
                                                                                    hr09s - float64
                                                                                    loan_amnt - float64
                                                                                    rev401 - float64
                                                                                    debt_to_income - float64
                                                                                    g099s - float64
                                                                                    all803 - float64
                                                                                    st27s - float64
                                                                                    g960s - float64
                                                                                    g230s - float64
                                                                                    all807 - float64
                                                                                    mt47s - float64
                                                                                    g094s - float64
                                                                                    all801 - float64
                                                                                    g104s - float64
                                                                                    s004s - float64
                                                                                    funding_threshold - float64
                                                                                    hi57s - float64
                                                                                    rep001 - float64
                                                                                    all301 - float64
                                                                                    rev302 - float64
                                                                                    hr12s - float64
                                                                                    bc102s - float64
                                                                                    iln740 - float64
                                                                                    all780 - float64
                                                                                    rep901 - float64
                                                                                    in36s - float64
                                                                                    mt36s - float64
                                                                                    iln724 - float64
                                                                                    aut720 - float64
                                                                                    rev703 - float64
                                                                                    inquiries_last6_months - float64
                                                                                    ale724 - float64
                                                                                    rep071 - float64
                                                                                    ale720 - float64
                                                                                    total_inquiries - float64
                                                                                    re102s - float64
                                                                                    bac302 - float64
                                                                                    credit_score - float64
                                                                                    int_rate - float64
                                                                                    monthly_pmt - float64
                                                                                    occupation - object
                                                                                    borrower_city - object
                                                                                    co_amount - float64
                                                                                    issue_date - datetime64[ns]
        :param y: binary ndarray of shape (num_samples,), the target (Charge off or no charge off)
        :return: self
        """
        self._max_date = X['issue_date'].max()
        X = pd.concat([X, pd.Series(y, name='CO')], axis=1)
        X = self._preprocess(X)
        X = X.drop(columns=['CO'])

        self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts charge-off for given loans.
        The model is capable of predicting only new loans, means the issue_date must be later than the latest issue_date
        in the training data.
        :param X: Pandas DataFrame of loans data, containing the following columns: all804 - float64
                                                                                    rep501 - float64
                                                                                    hr24s - float64
                                                                                    rt24s - float64
                                                                                    iln703 - float64
                                                                                    hr06s - float64
                                                                                    g099a - float64
                                                                                    iln720 - float64
                                                                                    percent_funded - float64
                                                                                    hr09s - float64
                                                                                    loan_amnt - float64
                                                                                    rev401 - float64
                                                                                    debt_to_income - float64
                                                                                    g099s - float64
                                                                                    all803 - float64
                                                                                    st27s - float64
                                                                                    g960s - float64
                                                                                    g230s - float64
                                                                                    all807 - float64
                                                                                    mt47s - float64
                                                                                    g094s - float64
                                                                                    all801 - float64
                                                                                    g104s - float64
                                                                                    s004s - float64
                                                                                    funding_threshold - float64
                                                                                    hi57s - float64
                                                                                    rep001 - float64
                                                                                    all301 - float64
                                                                                    rev302 - float64
                                                                                    hr12s - float64
                                                                                    bc102s - float64
                                                                                    iln740 - float64
                                                                                    all780 - float64
                                                                                    rep901 - float64
                                                                                    in36s - float64
                                                                                    mt36s - float64
                                                                                    iln724 - float64
                                                                                    aut720 - float64
                                                                                    rev703 - float64
                                                                                    inquiries_last6_months - float64
                                                                                    ale724 - float64
                                                                                    rep071 - float64
                                                                                    ale720 - float64
                                                                                    total_inquiries - float64
                                                                                    re102s - float64
                                                                                    bac302 - float64
                                                                                    credit_score - float64
                                                                                    int_rate - float64
                                                                                    monthly_pmt - float64
                                                                                    occupation - object
                                                                                    borrower_city - object
                                                                                    co_amount - float64
                                                                                    issue_date - datetime64[ns]
        :return: ndarray of shape (X.shape[0],), a binary array with the charge-off predictions
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= BEST_THRESHOLD).astype(int)

    def predict_proba(self, X):
        """
        Predicts charge-off probabilities for given loans
        :param X: Pandas DataFrame of loans data, containing the following columns: all804 - float64
                                                                                    rep501 - float64
                                                                                    hr24s - float64
                                                                                    rt24s - float64
                                                                                    iln703 - float64
                                                                                    hr06s - float64
                                                                                    g099a - float64
                                                                                    iln720 - float64
                                                                                    percent_funded - float64
                                                                                    hr09s - float64
                                                                                    loan_amnt - float64
                                                                                    rev401 - float64
                                                                                    debt_to_income - float64
                                                                                    g099s - float64
                                                                                    all803 - float64
                                                                                    st27s - float64
                                                                                    g960s - float64
                                                                                    g230s - float64
                                                                                    all807 - float64
                                                                                    mt47s - float64
                                                                                    g094s - float64
                                                                                    all801 - float64
                                                                                    g104s - float64
                                                                                    s004s - float64
                                                                                    funding_threshold - float64
                                                                                    hi57s - float64
                                                                                    rep001 - float64
                                                                                    all301 - float64
                                                                                    rev302 - float64
                                                                                    hr12s - float64
                                                                                    bc102s - float64
                                                                                    iln740 - float64
                                                                                    all780 - float64
                                                                                    rep901 - float64
                                                                                    in36s - float64
                                                                                    mt36s - float64
                                                                                    iln724 - float64
                                                                                    aut720 - float64
                                                                                    rev703 - float64
                                                                                    inquiries_last6_months - float64
                                                                                    ale724 - float64
                                                                                    rep071 - float64
                                                                                    ale720 - float64
                                                                                    total_inquiries - float64
                                                                                    re102s - float64
                                                                                    bac302 - float64
                                                                                    credit_score - float64
                                                                                    int_rate - float64
                                                                                    monthly_pmt - float64
                                                                                    occupation - object
                                                                                    borrower_city - object
                                                                                    co_amount - float64
                                                                                    issue_date - datetime64[ns]
        :return: ndarray of shape (X.shape[0], 2), contains the charge-off predictions probabilities (1-p, p)
        """
        if X['issue_date'].min() <= self._max_date:
            raise ValueError(f"issue_date must be later than latest fitted date, latest fitted date: {self._max_date}")

        X = self._preprocess(X, train=False)
        return self._model.predict_proba(X)

