import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from models.trans_prob import fit_pipeline as fit_trans_prob_pipeline
from models.transformers import BinaryEncoder, CyclicEncoder


class ModelingTable(TransformerMixin, BaseEstimator):

    def __init__(self, prediction_table):
        self.prediction_table = prediction_table

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y=None):
        # Prepare the affiliate and hotel country encoder, support unknown affiliates (-1)
        self.affiliate_encoder_ = LabelEncoder().fit([-1] + X['affiliate_id'].tolist())
        assert self.affiliate_encoder_.classes_[0] == -1

        self.hotel_country_encoder_ = LabelEncoder().fit([-1] + X['hotel_country'].tolist())
        assert self.hotel_country_encoder_.classes_[0] == '-1'

        self.booker_country_encoder_ = LabelEncoder().fit(X['booker_country'].tolist())

        return self

    # noinspection PyPep8Naming
    def transform(self, features: pd.DataFrame):
        # Encode affiliate_ids and hotel_country
        known_affiliates = set(self.affiliate_encoder_.classes_)
        known_hotel_countries = set(self.hotel_country_encoder_.classes_)

        data = features.copy()

        # Replace unknowns and encode
        data.loc[:, 'affiliate_id'] = data['affiliate_id'].apply(
            lambda affiliate_id: affiliate_id if affiliate_id in known_affiliates else -1)
        data.loc[:, 'affiliate_id'] = self.affiliate_encoder_.transform(data['affiliate_id'])
        data.loc[:, 'hotel_country'] = data['hotel_country'].apply(
            lambda hotel_country: hotel_country if hotel_country in known_hotel_countries else '-1')
        data.loc[:, 'hotel_country'] = self.hotel_country_encoder_.transform(data['hotel_country'])
        data.loc[:, 'booker_country'] = self.booker_country_encoder_.transform(data['booker_country'])

        # Reorder for relevant features
        data = data.groupby('utrip_id', sort=False).agg({
            'city_id': ['first', 'last'],
            'booker_country': 'last',
            'affiliate_id': 'last',
            'hotel_country': ['first', 'last'],
            'checkin': 'first',
            'checkout': 'last'}).reset_index()
        data.columns = ['_'.join(col).rstrip('_') for col in data.columns.values]  # Flatten column names

        # Create features - week numbers
        data['first_week_number'] = data['checkin_first'].dt.isocalendar().week - 1  # -1 to go from 0 to 52, this is
        data['last_week_number'] = data['checkout_last'].dt.isocalendar().week - 1  # relevant for cyclic encoder

        # Prepare top 3 destinations for binarization
        data['most_probable_city_0'] = data['city_id_last'].apply(lambda x: self.prediction_table[x, 0])
        data['most_probable_city_1'] = data['city_id_last'].apply(lambda x: self.prediction_table[x, 1])
        data['most_probable_city_2'] = data['city_id_last'].apply(lambda x: self.prediction_table[x, 2])

        # Drop irrelevant features
        data = data.drop(columns=['checkin_first', 'checkout_last', 'utrip_id'])

        return data


# noinspection PyUnusedLocal
def fit_pipeline(features: pd.DataFrame, labels: pd.DataFrame, n_cities: int, **kwargs) -> Pipeline:
    """
    Create and fit pipeline

    :param features: Features, should include the entire raw data format, cities already encoded
    :param labels: Labels, should include only 'utrip_id' and 'city_id' columns with already encoded cities
    :param n_cities: Number of unique cities in the features
    :param kwargs: Support additional arguments for fitting
    :return: Fitted pipeline
    """

    # Use the benchmark model to get the prediction table
    # where prediction_table[i,j] is the jth most probable city to travel to from i (0 is most probable)
    prediction_table = fit_trans_prob_pipeline(features, labels).steps[-1][1].prediction_table_

    # Prerequisites
    n_affiliates = features.affiliate_id.nunique() + 1  # +1 for unknown
    n_hotel_countries = features.hotel_country.nunique() + 1
    n_booker_countries = features.booker_country.nunique()

    # Build pipeline
    pipeline = Pipeline([('modeling_table', ModelingTable(prediction_table=prediction_table)),
                         ('binarize_features', ColumnTransformer([('binarizer',
                                                                   BinaryEncoder(
                                                                       [n_cities, n_cities, n_cities, n_cities,
                                                                        n_cities, n_booker_countries, n_affiliates,
                                                                        n_hotel_countries, n_hotel_countries]),
                                                                   ['city_id_first',
                                                                    'city_id_last',
                                                                    'most_probable_city_0',
                                                                    'most_probable_city_1',
                                                                    'most_probable_city_2',
                                                                    'booker_country_last',
                                                                    'affiliate_id_last',
                                                                    'hotel_country_first',
                                                                    'hotel_country_last']),
                                                                  ('cyclic_encoder',
                                                                   CyclicEncoder(maximum=[52, 52]),
                                                                   ['first_week_number', 'last_week_number'])],
                                                                 remainder='drop')),
                         ('xgboost', XGBClassifier())])
    # Fit
    pipeline.fit(features, labels['city_id'].values)

    return pipeline
