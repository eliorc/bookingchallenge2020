import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from models.trans_prob import fit_pipeline as fit_trans_prob_pipeline
from models.transformers import BinaryEncoder


class ModelingTable(TransformerMixin, BaseEstimator):

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y=None):
        # Prepare the affiliate encoder, support unknown affiliates (-1)
        self.affiliate_encoder = LabelEncoder().fit([-1] + X['affiliate_id'].tolist())
        assert self.affiliate_encoder.classes_[0] == -1

        # Use the benchmark model to get the prediction table
        # where prediction_table[i,j] is the jth most probable city to travel to from i (0 is most probable)
        self.prediction_table = fit_trans_prob_pipeline(X, y).steps[-1][1].prediction_table_

        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        # Encode affiliate_ids
        known_affiliates = set(self.affiliate_encoder.classes_)
        X.loc[:, 'affiliate_id'] = X['affiliate_id'].apply(
            lambda affiliate_id: affiliate_id if affiliate_id in known_affiliates else -1)
        X.loc[:, 'affiliate_id'] = self.affiliate_encoder.transform(X['affiliate_id'])

        # Reorder for relevant features
        X = X.groupby('utrip_id', sort=False).agg({
            'city_id': ['first', 'last'],
            'booker_country': 'last',
            'affiliate_id': 'last',
            'hotel_country': ['first', 'last'],
            'checkin': 'first',
            'checkout': 'last'}).reset_index()
        X.columns = ['_'.join(col).rstrip('_') for col in X.columns.values]  # Flatten column names

        # Create features - week numbers
        X['first_week_number'] = X['checkin_first'].dt.isocalendar().week
        X['last_week_number'] = X['checkout_last'].dt.isocalendar().week

        # Prepare top 3 destinations for binarization
        X['most_probable_city_0'] = X['city_id_last'].apply(lambda x: self.prediction_table[x, 0])
        X['most_probable_city_1'] = X['city_id_last'].apply(lambda x: self.prediction_table[x, 1])
        X['most_probable_city_2'] = X['city_id_last'].apply(lambda x: self.prediction_table[x, 2])

        # Drop irrelevant features
        X = X.drop(columns=['checkin_first', 'checkout_last', 'utrip_id'])

        return X


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

    # Prerequisites
    n_affiliates = features.affiliate_id.nunique() + 1  # +1 for unknown
    n_booker_countries = features.booker_country.nunique()
    n_hotel_countries = features.hotel_country.nunique()

    # Build pipeline
    pipeline = Pipeline([('modeling_table', ModelingTable()),
                         ('binarize_features', ColumnTransformer([('binarizer',
                                                                   BinaryEncoder(
                                                                       [n_cities, n_cities, n_cities, n_cities,
                                                                        n_cities, n_booker_countries, n_affiliates,
                                                                        n_hotel_countries, n_hotel_countries],
                                                                       [False, False, False, False, False, True, True,
                                                                        True, True]),
                                                                   ['city_id_first',
                                                                    'city_id_last',
                                                                    'most_probable_city_0',
                                                                    'most_probable_city_1',
                                                                    'most_probable_city_2',
                                                                    'booker_country_last',
                                                                    'affiliate_id_last',
                                                                    'hotel_country_first',
                                                                    'hotel_country_last'],
                                                                   )], remainder='passthrough'))])
    # Fit
    pipeline.fit(features, labels['city_id'].values)

    return pipeline
