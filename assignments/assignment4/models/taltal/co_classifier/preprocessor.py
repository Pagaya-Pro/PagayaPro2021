from sklearn import preprocessing


class Cleandata:

    def fit(self, X, y):
        X = X.copy()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(X.issue_date)
        X['co'] = y
        self.occupation_to_co = X.groupby('occupation')['co'].mean()
        X = X.drop('co', axis=1)
        self.to_drop = set()
        df = X.corr().abs().stack()
        for tup in df[(df > .7) & (df < 1)].index:
            if not set(tup) & self.to_drop:
                self.to_drop.add(tup[0])

    def transform(self, X):
        X = X.copy()
        X = X.drop('borrower_city', axis=1)
        X['issue_date'] = self.le.transform(X.issue_date)
        X.occupation.fillna('Nan', inplace=True)
        X = X.join(self.occupation_to_co, on='occupation', rsuffix='_occupation') \
            .drop('occupation', axis=1).rename({'co_occupation': 'occupation_pr'})
        X = X.drop('co', axis=1)
        X = X.drop(list(self.to_drop), axis=1)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


