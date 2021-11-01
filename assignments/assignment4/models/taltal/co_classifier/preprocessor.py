from sklearn import preprocessing

class Cleandata:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(X.issue_date)
        self.occupation_to_co = X.groupby('occupation')['co'].mean()
        self.to_drop = set()
        for tup in df[(df > .7) & (df < 1)].index:
            if not set(tup) & to_drop:
                self.to_drop.add(tup[0])

    def transform(self, X):
        X = X.copy()
        X = X.drop('borrower_city', axis=1)
        X['issue_date'] = self.le.transform(X.issue_date)
        X.occupation.fillna('Nan', inplace=True)
        X = X.join(self.occupation_to_co, on='occupation', rsuffix='_occupation') \
            .drop('occupation', axis=1).rename({'co_occupation': 'occupation_pr'})
        X = X.drop(list(to_drop), axis=1)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

        # X = X.drop('borrower_city', axis=1)
        # self.le = preprocessing.LabelEncoder()
        # X['issue_date'] = self.le.fit_transform(X.issue_date)
        # X.occupation.fillna('Nan', inplace=True)
        # self.occupation_to_co = X.groupby('occupation')['co'].mean()
        # X = X.join(self.occupation_to_co, on='occupation', rsuffix='_occupation') \
        #     .drop('occupation', axis=1).rename({'co_occupation': 'occupation_pr'})
        # self.to_drop = set()
        # for tup in df[(df > .7) & (df < 1)].index:
        #     if not set(tup) & to_drop:
        #         self.to_drop.add(tup[0])
        # X = X.drop(list(to_drop), axis=1)

