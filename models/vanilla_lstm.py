from collections import Counter, defaultdict
from typing import Dict

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from models.transformers import CyclicEncoder


class ModelingTable(TransformerMixin, BaseEstimator):
    """
    Converts checkin/checkout to week numbers, and drops user IDs
    """

    # noinspection PyUnusedLocal
    def fit(self, features, labels=None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, features):
        data = features.copy()

        # Convert checkins to week numbers
        data.loc[:, 'checkin'] = data.loc[:, 'checkin'].dt.isocalendar().week
        data.loc[:, 'checkout'] = data.loc[:, 'checkout'].dt.isocalendar().week

        # Drop user ID
        data = data.drop(columns=['user_id'])

        return data


class TopNDestinations(TransformerMixin, BaseEstimator):

    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    # noinspection PyAttributeOutsideInit
    def fit(self, features, labels=None):

        # Get transition counter and prior
        transition_counter = self.create_transition_counter(features, labels)

        # Compute prior counter (target cities counter
        prior_counter = sum(transition_counter.values(), Counter())

        # Get top 3 per source
        self.top_n_mapping_ = dict()
        for source in transition_counter:

            self.top_n_mapping_[source] = [target for target, _ in transition_counter[source].most_common(n=self.top_n)]
            # Fill missing with prior
            if (n_targets := len(self.top_n_mapping_[source])) < self.top_n:
                gap = self.top_n - n_targets
                self.top_n_mapping_[source] += [target for target, _ in prior_counter.most_common(n=gap)]

        # Save top n prior for unknown sources
        self.top_n_prior_ = [target for target, _ in prior_counter.most_common(n=self.top_n)]

        return self

    def transform(self, features: pd.DataFrame):

        # Add top_n features of the most probable cities
        for n in range(self.top_n):
            features[f'{n + 1}_most_probable_city_id'] = features.groupby('utrip_id')['city_id'].transform(
                lambda x: self.top_n_mapping_.get(x.iat[-1], self.top_n_prior_)[n])

        return features

    @staticmethod
    def create_transition_counter(features: pd.DataFrame, labels: pd.DataFrame) -> Dict[int, Counter]:
        """
        Create a counter where counter[i][j] holds the counts the number of times travelled between i and j.
        Since each counter[i] is an collections.Counter object, it has ``most_common()`` method to easily access the
        most common cities to travel to.

        :param features: Data from "fit"
        :param labels: Labels from fit
        :return: Transition counter
        """

        source_cities = features.groupby('utrip_id', sort=False)['city_id'].last().values
        target_cities = labels['city_id'].values

        assert len(source_cities) == len(target_cities)

        transition_counter = defaultdict(Counter)
        for source, target in zip(source_cities, target_cities):
            transition_counter[source][target] += 1

        return transition_counter


def fit_pipeline(top_n_targets: int = 3, **kwargs) -> Pipeline:
    """
    Create and fit pipeline

    :param top_n_targets: Top targets to include as features
    :param kwargs: Additional params
    :return: Fitted pipeline
    """

    # Use the benchmark model to get the prediction table
    # where prediction_table[i,j] is the jth most probable city to travel to from i (0 is most probable)
    # Preprocess
    top_n_targets_columns = [f'{n + 1}_most_probable_city_id' for n in range(top_n_targets)]
    preprocess = Pipeline(
        [('top_n_targets', TopNDestinations(top_n_targets)),  # Step 1 - create top target cities feature
         ('modeling_table',  # Step 2 - create features and prepare data
          ModelingTable()),
         ('encode',  # Step 3 - ordinal encode categories, and cyclic encode week numbers
          ColumnTransformer([('cyclic', CyclicEncoder(), ['checkin',
                                                          'checkout']),
                             ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                        unknown_value=-1),
                              ['city_id',
                               'device_class',
                               'affiliate_id',
                               'booker_country',
                               'hotel_country'] + top_n_targets_columns)],
                            remainder='passthrough')),
         ('unknown2zero',  # Step 4 - Convert unknowns from -1 to 0, by adding 1
          ColumnTransformer([('add_1',
                              FunctionTransformer(lambda x: x + 1), [2,  # City ID
                                                                     4,  # Affiliate
                                                                     6])],  # Hotel country
                            remainder='passthrough')),
         ('to_df',
          FunctionTransformer(lambda x: pd.DataFrame(x, columns=['checkin_sin', 'checkin_cos',
                                                                 'checkout_sin', 'checkout_out',
                                                                 'city_id',
                                                                 'device_class',
                                                                 'affiliate_id',
                                                                 'booker_country',
                                                                 'hotel_country'] + top_n_targets_columns + [
                                                                    'utrip_id'])))])

    return preprocess
