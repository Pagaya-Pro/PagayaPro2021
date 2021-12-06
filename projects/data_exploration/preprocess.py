def preprocess_data(X,y=None, na_threshold = 0.6):
    """
    preprocess for fitting and predicting on the model
    :param X: features
    :param y: labels, optional for prediction
    :param na_threshold: optional threshold for fropping nan columns
    :return:
    """
    df = X.assign(label=y) if y is not None else X
    preprocessed_df = (
        df
        .assign(loan_amnt = pd.to_numeric(df.loan_amnt, errors="coerce"),
                eoddate = pd.to_datetime(df.eoddate, errors="coerce")
               )
        .dropna(subset=["int_rate",'label'])
        .select_dtypes(include="number")
        .dropna(axis=1, how="all", thresh=int(na_threshold * df.shape[0]))
    )
    if y is None:
        return preprocessed_df
    return preprocessed_df.drop(columns=['label']), preprocessed_df['label']