import pandas as pd
import numpy as np
import pickle
import swifter
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely.wkt
from shapely.validation import make_valid
from shapely.ops import unary_union
from datetime import datetime
import xgboost as xgb


class Tamy:
    """
    A class that implements our AVM model TAMY (Tal, Amit, Maoz, Yuval)
    """

    def __init__(self, add_neighborhood_dummies=True, index_path='BMN_indices_up_to_2021_06_01.parquet',
                 params_path='best_params.pkl'):
        """
        Constructor
        :param add_neighborhood_dummies: Add dummies for each neighborhood in the train set.
        :param index_path: Path to an index file.
        :param params_path: Path to an XGBoost params file.
        """
        self.train_size = 0
        self.fea_dict = {}
        self.train_columns = None
        self.add_neighborhood_dummies = add_neighborhood_dummies
        self.index = pd.read_parquet(index_path)
        self.model = None
        self.train_mode = True

        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)

        self.columns = pd.Series(['SaleDate', 'APNSeqNbr', 'OldAPN', 'OldApnIndicator',
                                  'SitusLatitude', 'SitusLongitude', 'SitusGeoStatusCode', 'PropertyClassID',
                                  'LandUseCode', 'StateLandUseCode', 'CountyLandUseCode', 'Zoning',
                                  'MobileHomeInd', 'TimeshareCode', 'LotSizeFrontageFeet', 'LotSizeDepthFeet',
                                  'LotSizeAcres', 'LotSizeSqFt', 'Owner1CorpInd', 'Owner1Suffix',
                                  'Owner2CorpInd', 'Owner2Suffix', 'OwnerOccupied', 'Owner1OwnershipRights',
                                  'MailingFullStreetAddress', 'MailingHouseNbrSuffix', 'MailingOptOut',
                                  'AssdTotalValue', 'AssdLandValue', 'AssdImprovementValue',
                                  'MarketTotalValue', 'MarketValueLand', 'MarketValueImprovement', 'TaxAmt',
                                  'TaxYear', 'TaxDeliquentYear', 'MarketYear', 'AssdYear', 'TaxRateCodeArea',
                                  'SchoolTaxDistrict1Code', 'SchoolTaxDistrict2Code',
                                  'SchoolTaxDistrict3Code', 'HomesteadInd', 'VeteranInd', 'DisabledInd',
                                  'WidowInd', 'SeniorInd', 'SchoolCollegeInd', 'ReligiousInd', 'WelfareInd',
                                  'PublicUtilityInd', 'CemeteryInd', 'HospitalInd', 'LibraryInd',
                                  'BuildingArea', 'BuildingAreaInd', 'SumBuildingSqFt', 'SumLivingAreaSqFt',
                                  'SumGrossAreaSqFt', 'SumAdjAreaSqFt', 'SumBasementSqFt', 'SumGarageSqFt',
                                  'YearBuilt', 'EffectiveYearBuilt', 'Bedrooms', 'TotalRooms', 'BathTotalCalc',
                                  'BathFull', 'BathsPartialNbr', 'BathFixturesNbr', 'Amenities',
                                  'AirConditioningCode', 'BasementCode', 'BuildingClassCode',
                                  'BuildingConditionCode', 'ConstructionTypeCode', 'DeckInd',
                                  'ExteriorWallsCode', 'InteriorWallsCode', 'FireplaceCode', 'FloorCoverCode',
                                  'Garage', 'HeatCode', 'HeatingFuelTypeCode', 'SiteInfluenceCode',
                                  'GarageParkingNbr', 'DrivewayCode', 'OtherRooms', 'PatioCode', 'PoolCode',
                                  'PorchCode', 'BuildingQualityCode', 'RoofCoverCode', 'RoofTypeCode',
                                  'SewerCode', 'StoriesNbrCode', 'StyleCode', 'SumResidentialUnits',
                                  'SumBuildingsNbr', 'SumCommercialUnits', 'TopographyCode', 'WaterCode',
                                  'AssessorsMapRef', 'ProviderTimeStamp'])

    def check_validity(self, X: pd.DataFrame, y: pd.Series):
        """
        Checks that the given train set is valid.
        """
        assert (len(X) == len(y))
        assert (np.sum(y.isna()) == 0)
        assert np.all(self.columns.isin(X.columns))

    def train_one_hot_encoding(self, df):
        """
        Adds dummy varaibales for columns.
        """
        print('One hot encoding started')

        # editing pool feature
        df.loc[df['PoolCode'] == 3, 'PoolCode'] = np.nan
        df.PoolCode = df.PoolCode.fillna(0)
        df.loc[(df['PoolCode'] != 0) & (df['PoolCode'] != 9), 'PoolCode'] = 8

        col_to_encode = ['SumResidentialUnits',
                         'AirConditioningCode',
                         'PoolCode',
                         'HeatCode',
                         'RoofCoverCode',
                         'RoofTypeCode',
                         'StyleCode']

        if self.add_neighborhood_dummies:
            col_to_encode += ['LA_Neighborhood']

        for col in col_to_encode:
            # Get one hot encoding of col
            one_hot = pd.get_dummies(df[col], drop_first=False)
            # Change columns' names
            for i in one_hot.columns:
                one_hot = one_hot.rename(columns={i: f'{col}{i}'})
            # drop original column
            df = df.drop(col, axis=1)
            # Join the encoded df
            df = pd.concat([df, one_hot], axis=1)
        print('One hot encoding finished.')

        return df

    def fill_property_size(self, df):
        """
        Fill property size (according to the logic in the notion)
        """
        df = df.copy()

        df.LotSizeFrontageFeet = pd.to_numeric(df.LotSizeFrontageFeet)

        df.loc[df.LotSizeFrontageFeet < 150, 'LotSizeFrontageFeet'] = np.nan
        df.loc[df.LotSizeDepthFeet < 150, 'LotSizeDepthFeet'] = np.nan
        df.loc[df.LotSizeSqFt < 1000, 'LotSizeSqFt'] = np.nan

        # fixing order of magnitude
        df.LotSizeFrontageFeet = df.LotSizeFrontageFeet / 10
        df.LotSizeDepthFeet = df.LotSizeDepthFeet / 10

        # filling nans
        df.LotSizeFrontageFeet = df.LotSizeFrontageFeet.fillna(df.LotSizeSqFt / df.LotSizeDepthFeet)
        df.LotSizeDepthFeet = df.LotSizeDepthFeet.fillna(df.LotSizeSqFt / df.LotSizeFrontageFeet)
        df.LotSizeSqFt = df.LotSizeSqFt.fillna(df.LotSizeDepthFeet * df.LotSizeFrontageFeet)

        # if SizeFront/SizeDepth have nans
        df.LotSizeFrontageFeet = df.LotSizeFrontageFeet.fillna(np.sqrt(df.LotSizeSqFt))
        df.LotSizeDepthFeet = df.LotSizeDepthFeet.fillna(np.sqrt(df.LotSizeSqFt))

        k = len(df)
        bad_ids = df[(df.LotSizeSqFt.isna()) | (df.LotSizeFrontageFeet.isna())].index
        df = df[(df.LotSizeSqFt.notna()) | (df.LotSizeFrontageFeet.notna())]
        print(f'we dropped {k - len(df)} rows with nan size.')

        if (not self.train_mode) and (len(bad_ids) > 0):
            print(f'Had bad house dimentions in {bad_ids}')
            raise ValueError(f'Had bad size in {bad_ids}')

        return df

    @staticmethod
    def point_isin(point, neigh):
        """input: SitusLongitude and SitusLatitude of a house and list of LA neighborhoods as polygons.
        output: returns the name of the neighborhood of the house."""
        for i in range(len(neigh)):
            if neigh.polygon[i].contains(point):
                return neigh.name[i]

    @staticmethod
    def get_neighborhood(df, la_neigh):
        """ applies point_isin on all the rows of the dataframe"""
        df = df.reset_index()

        df['LA_Neighborhood'] = df.swifter.apply(
            lambda x: Tamy.point_isin(Point(x.SitusLongitude, x.SitusLatitude), la_neigh), axis=1)
        df = df.set_index('PropertyID', drop=True)
        return df

    @staticmethod
    def add_neighborhood(combined):
        """ Add neighborhood for each property """
        with open(f'la_neigh.pkl', 'rb') as f:
            la_neigh = pickle.load(f)

        combined['SitusLongitude'] = pd.to_numeric(combined.SitusLongitude)
        combined['SitusLatitude'] = pd.to_numeric(combined.SitusLatitude)

        combined = Tamy.get_neighborhood(combined, la_neigh)

        return combined

    def fix_neighborhoods(self, combined):
        """ Fix neighborhoods """
        property_groups = combined.groupby('PropertyID').last()
        combined = combined.reset_index()
        combined['LA_Neighborhood'] = (
            combined.swifter.apply(lambda x: property_groups.loc[x.PropertyID, 'LA_Neighborhood'], axis=1)).values
        combined = combined.set_index('PropertyID', drop=True)

        print('remove houses from suburbs.')
        l = len(combined)
        bad_neighborhoods = ['Griffith Park', 'Hansen Dam', 'Sepulveda Basin', 'South Diamond Bar', 'Universal City',
                             'Wittier Narrows', 'Whittier Narrows']
        bad_ids = combined[combined.LA_Neighborhood.isin(bad_neighborhoods)].index
        combined = combined[~combined.LA_Neighborhood.isin(bad_neighborhoods)]
        print(f'dropped {l - len(combined)} houses that appears in irrelevant neighborhoods.')

        if (not self.train_mode) and (len(bad_ids) > 0):
            print(f'Had bad coordinates in {bad_ids}.')
            raise ValueError(f'Had bad coordinates in {bad_ids}')

        combined.loc[combined.LA_Neighborhood == 'Chatsworth Reservoir', 'LA_Neighborhood'] = 'Chatsworth'

        return combined

    def create_features(self, X: pd.DataFrame):
        """ Create and edit new features for our train set. """
        print('Create features started')

        # Add spatial feature
        X = Tamy.add_neighborhood(X)
        X = self.fix_neighborhoods(X)

        # Add temporal feature
        X['month_in_year'] = X.SaleDate.apply(lambda x: x.month)
        X['year'] = X.SaleDate.apply(lambda x: x.year)

        # Add flag for isna
        X['BuildingConditionCodeIsna'] = X['BuildingConditionCode'].isna()
        X['BuildingQualityCodeIsna'] = X['BuildingQualityCode'].isna()
        X['StoriesNbrCodeIsna'] = X['StoriesNbrCode'].isna()

        # changing basement feature to boolean.
        X[['FireplaceCode', 'Garage']] = X[['FireplaceCode', 'Garage']].notna().astype(int)

        # Numeric
        X = self.fill_property_size(X)
        X['SitusLongitude'] = pd.to_numeric(X.SitusLongitude)
        X['SitusLatitude'] = pd.to_numeric(X.SitusLatitude)

        # filling missing values
        fill_0 = ['AirConditioningCode', 'HeatCode', 'RoofCoverCode', 'RoofTypeCode', 'StyleCode']
        X[fill_0] = X[fill_0].fillna(0)

        X.loc[(X['StyleCode'] != 15) | (X['StyleCode'] != 0), 'StyleCode'] = -15
        X.loc[X['HeatCode'] == 24, 'HeatCode'] = 0
        X.loc[X['RoofCoverCode'].isin([1, 2, 8]), 'RoofCoverCode'] = 0

        # Encoding categorical
        X = self.train_one_hot_encoding(X)

        if self.train_columns is not None:
            # Add columns from train that are missing in the test. Fill the added columns with 0
            for col in self.train_columns:
                if col not in X.columns:
                    X[col] = 0
        else:
            self.fea_dict['BuildingConditionCode'] = X['BuildingConditionCode'].median()
            self.fea_dict['BuildingQualityCode'] = X['BuildingQualityCode'].median()
            self.fea_dict['StoriesNbrCode'] = X['StoriesNbrCode'].median()
            self.fea_dict['EffectiveYearBuilt'] = X['EffectiveYearBuilt'].median()

        X['BuildingConditionCode'] = X['BuildingConditionCode'].fillna(self.fea_dict['BuildingConditionCode'])
        X['BuildingQualityCode'] = X['BuildingQualityCode'].fillna(self.fea_dict['BuildingQualityCode'])
        X['StoriesNbrCode'] = X['StoriesNbrCode'].fillna(self.fea_dict['StoriesNbrCode'])
        X['EffectiveYearBuilt'] = X['EffectiveYearBuilt'].fillna(self.fea_dict['EffectiveYearBuilt'])

        print('Create features finished.')
        return X

    def drop_correlated(self, df):
        """
        drops features that are correlation values greater than 0.75 with more than 2 other features
        """
        print('drop correlated features')
        cor_mat = df.corr()
        for i, j in enumerate(range(len(cor_mat))):
            if i == j:
                cor_mat.iloc[i, j] = np.nan
        while True:
            max_correlated = cor_mat[cor_mat > 0.75].count().idxmax()
            if cor_mat[cor_mat > 0.75].count()[max_correlated] >= 2:
                cor_mat = cor_mat.drop(max_correlated, axis=0).drop(max_correlated, axis=1)
            else:
                break
        columns_dropped = set(df.columns) - set(cor_mat.columns)
        return columns_dropped

    def remove_unicolumn(self, df):
        """removes columns with only one value"""
        column_to_drop = []
        for col in df.columns:
            n_categories = len(df[col].unique())
            if n_categories == 1:
                column_to_drop.append(col)
        df = df.drop(column_to_drop, axis=1)
        print(f'found and dropped {len(column_to_drop)} unicolumns.')
        return df

    def drop_features_test(self, df):
        """ """
        df = df[self.train_columns]
        print(f'eventually had {df.isna().sum().sum()} rows with unexpected NaNs in the test/val. dropped those rows')
        print('This is a df with the bad properties:', df[df.isna().sum(axis=1) > 0])
        # ToDo - handle with unexpected NaNs
        return df

    def drop_features_train(self, df):
        """ Drop features from train. """
        # dropping features
        features_to_drop = ['SitusGeoStatusCode',
                            'TotalRooms',
                            'LandUseCode',
                            'MobileHomeInd',
                            'AssdYear',
                            'SewerCode',
                            'LotSizeAcres',
                            'AssdTotalValue',
                            'YearBuilt',
                            'BathTotalCalc',
                            'BathFixturesNbr',
                            'BasementCode',
                            'SumCommercialUnits',
                            'WaterCode',
                            'HeatingFuelTypeCode',
                            'SumBuildingsNbr',
                            'TaxDeliquentYear',
                            'LotSizeDepthFeet',
                            'SumLivingAreaSqFt',
                            'TaxAmt',
                            'SumBasementSqFt',
                            'SumGarageSqFt',
                            'ExteriorWallsCode',
                            'BuildingClassCode',
                            'ConstructionTypeCode',
                            'SumGrossAreaSqFt',
                            'SumBuildingSqFt',
                            'AssdLandValue',
                            'AssdImprovementValue',
                            'SitusLongitude',
                            'SitusLatitude']

        for fea in features_to_drop:
            df = df.drop(fea, axis=1)

        # removing all columns with type object
        non_numeric = ['APNSeqNbr', 'OldAPN', 'OldApnIndicator', 'PropertyClassID',
                       'StateLandUseCode', 'CountyLandUseCode', 'Zoning', 'TimeshareCode',
                       'Owner1CorpInd', 'Owner1Suffix', 'Owner2CorpInd', 'Owner2Suffix',
                       'OwnerOccupied', 'Owner1OwnershipRights', 'MailingFullStreetAddress',
                       'MailingHouseNbrSuffix', 'MailingOptOut', 'TaxRateCodeArea',
                       'SchoolTaxDistrict1Code', 'SchoolTaxDistrict2Code',
                       'SchoolTaxDistrict3Code', 'HomesteadInd', 'VeteranInd', 'DisabledInd',
                       'WidowInd', 'SeniorInd', 'SchoolCollegeInd', 'ReligiousInd',
                       'WelfareInd', 'PublicUtilityInd', 'CemeteryInd', 'HospitalInd',
                       'LibraryInd', 'BuildingAreaInd', 'Amenities', 'SiteInfluenceCode',
                       'OtherRooms', 'TopographyCode', 'AssessorsMapRef']
        print(f'Dropping features of type object')
        df = df.drop(non_numeric, axis=1)

        df = self.remove_unicolumn(df)

        columns_dropped = self.drop_correlated(df.drop('SaleAmt', axis=1).copy())
        df = df.drop(columns_dropped, axis=1)

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        :param X: Train set. Assumes that X has all assessor's columns and a SaleDate column. PropertyID is the index.
        :param y: Labels (sale amounts)
        :return: self
        """
        self.train_mode = True
        # initials
        self.check_validity(X, y)
        X = X[self.columns]
        X['SaleAmt'] = np.array(y)

        # update params
        self.train_size = len(X)

        # add features
        X = self.create_features(X)

        # drop features
        X = self.drop_features_train(X)

        y = X[['SaleAmt', 'month_in_year', 'year']].copy()
        X = X.drop(['SaleAmt'], axis=1)

        self.train_columns = X.columns

        X = X.drop(['month_in_year', 'year'], axis=1)

        y['month_sold'] = y.apply(lambda x: datetime(int(x.year), int(x.month_in_year), 1), axis=1)
        y = y.join(self.index, on='month_sold', how='left')
        y['SaleAmt'] = y.SaleAmt / y.BMN

        y = y['SaleAmt']

        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)

        return self

    def predict(self, X):
        """
        X - assumes that have assessor's columns, PropertyID is the index and SaleDate with the relevant SaleDate.
        If there are properties with unexpected nans - print them IDS.
        """
        self.train_mode = False

        assert np.all(self.columns.isin(X.columns))
        X = X[self.columns]

        # add features
        X = self.create_features(X)

        # drop features
        X = self.drop_features_test(X)

        ### In time predictions - need further analysis.

        #         y_times = X[['month_in_year', 'year']]
        #         X = X.drop(['month_in_year', 'year'], axis=1)
        #         dtest = xgb.DMatrix(X)

        #         y = self.model.predict(dtest)
        #         y_times['SaleAmt'] = y
        #         y_times['month_sold'] = y_times.apply(lambda x: datetime(int(x.year), int(x.month_in_year), 1), axis=1)

        #         y_times = y_times.join(self.index, on='month_sold', how='left')
        #         y_times['BMN'] = y_times['BMN'].fillna(1)
        #         print(y_times.BMN)

        #         y_times['SaleAmt'] = y_times.SaleAmt * y_times.BMN

        #         return np.array(y_times['SaleAmt'])

        X = X.drop(['month_in_year', 'year'], axis=1)
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


def run_model():

    with open(f'split_on_2020_02_01_short_train.pkl', 'rb') as f:
        train, val, test = pickle.load(f)

    label = train['SaleAmt']
    m = Tamy(index_path='BMN_indices_up_to_2020_02_01.parquet')

    m = m.fit(train.drop('SaleAmt', axis=1), label)

    a = m.predict(test.drop('SaleAmt', axis=1).drop(162647577, axis=0))

    ## MDAPE
    print(np.median(np.abs(np.array(test.drop([162647577])['SaleAmt'])-a) / np.array(test.drop([162647577])['SaleAmt'])))


