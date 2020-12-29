from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

from models.trans_prob import fit_pipeline as fit_trans_prob_pipeline


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
