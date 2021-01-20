from collections import Counter, defaultdict
from random import shuffle
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from models.transformers import CyclicEncoder


class VanillaLSTM(tf.keras.Model):

    def __init__(self,
                 n_cities: int,
                 n_target_cities: int,
                 n_devices: int,
                 n_affiliates: int,
                 n_booker_countries: int,
                 n_hotel_countries: int,
                 n_labels: int,
                 most_probable_cities_keys: List[str],
                 city_embedding_dim: int = 24,
                 target_city_embedding_dim: int = 24,
                 device_embedding_dim: int = 2,
                 affiliate_embedding_dim: int = 12,
                 booker_country_embedding_dim: int = 2,
                 hotel_country_embedding_dim: int = 6,
                 pre_lstm_projection_units: int = 48,
                 embedding_dropout_rate: float = 0.2,
                 lstm_units: int = 48,
                 bottleneck_units: int = 12,
                 fc_1_units: int = 48,
                 name: str = 'vanilla_lstm',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        # Self attributes
        self.n_cities = n_cities
        self.n_target_cities = n_target_cities
        self.n_devices = n_devices
        self.n_affiliates = n_affiliates
        self.n_booker_countries = n_booker_countries
        self.n_hotel_countries = n_hotel_countries
        self.n_labels = n_labels
        self.most_probable_cities_keys = most_probable_cities_keys
        self.city_embedding_dim = city_embedding_dim
        self.target_city_embedding_dim = target_city_embedding_dim
        self.device_embedding_dim = device_embedding_dim
        self.affiliate_embedding_dim = affiliate_embedding_dim
        self.booker_country_embedding_dim = booker_country_embedding_dim
        self.hotel_country_embedding_dim = hotel_country_embedding_dim
        self.embedding_dropout_rate = embedding_dropout_rate
        self.pre_lstm_projection_units = pre_lstm_projection_units
        self.lstm_units = lstm_units
        self.bottleneck_units = bottleneck_units
        self.fc_1_units = fc_1_units

        # Embedding matrices
        self.city_embeddings = tf.keras.layers.Embedding(input_dim=n_cities,
                                                         output_dim=city_embedding_dim,
                                                         mask_zero=True,
                                                         name='city_embeddings')
        self.city_emb_dropout = tf.keras.layers.SpatialDropout1D(rate=embedding_dropout_rate,
                                                                 name='city_embeddings_dropout')

        self.target_city_embeddings = tf.keras.layers.Embedding(input_dim=n_target_cities,
                                                                output_dim=target_city_embedding_dim,
                                                                mask_zero=False,
                                                                name='target_city_embeddings')

        self.device_embeddings = tf.keras.layers.Embedding(input_dim=n_devices,
                                                           output_dim=device_embedding_dim,
                                                           mask_zero=False,
                                                           name='device_embeddings')
        self.device_emb_dropout = tf.keras.layers.SpatialDropout1D(rate=embedding_dropout_rate,
                                                                   name='device_embeddings_dropout')

        self.affiliate_embeddings = tf.keras.layers.Embedding(input_dim=n_affiliates,
                                                              output_dim=affiliate_embedding_dim,
                                                              mask_zero=True,
                                                              name='affiliate_embeddings')
        self.affiliate_emb_dropout = tf.keras.layers.SpatialDropout1D(rate=embedding_dropout_rate,
                                                                      name='affiliate_embeddings_dropout')

        self.booker_country_embeddings = tf.keras.layers.Embedding(input_dim=n_booker_countries,
                                                                   output_dim=booker_country_embedding_dim,
                                                                   mask_zero=False,
                                                                   name='booker_country_embeddings')

        self.hotel_country_embeddings = tf.keras.layers.Embedding(input_dim=n_hotel_countries,
                                                                  output_dim=hotel_country_embedding_dim,
                                                                  mask_zero=True,
                                                                  name='hotel_country_embeddings')
        self.hotel_country_emb_dropout = tf.keras.layers.SpatialDropout1D(rate=embedding_dropout_rate,
                                                                          name='hotel_country_embeddings_dropout')

        # Projection, to project embeddings from different spaces to a singular one
        self.pre_lstm_projection = tf.keras.layers.Dense(units=pre_lstm_projection_units, activation=None)

        # LSTM
        self.lstm = tf.keras.layers.LSTM(units=lstm_units)

        # Post fully connected
        fc1_input_dim = lstm_units + booker_country_embedding_dim + int(
            target_city_embedding_dim * len(most_probable_cities_keys))
        self.fc1 = tf.keras.layers.Dense(units=fc_1_units, activation='relu')
        self.fc1.build(input_shape=tf.TensorShape([None, fc1_input_dim]))

        # Batch norm
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        # Bottleneck
        self.bottleneck = tf.keras.layers.Dense(units=bottleneck_units, activation='relu')

        # Output layer
        self.out = tf.keras.layers.Dense(units=n_labels, activation='softmax')

    def call(self, inputs, training: bool = False, **kwargs):
        # Compute mask
        mask = self.city_embeddings.compute_mask(inputs['city_id_x'])  # (batch_size, max_seq_len)

        # Embed categorical variables
        embedded = list()  # list of (batch_size, max_seq_len, emb_dim), emb_dim different for each
        embedded.append(self.city_emb_dropout(
            self.city_embeddings(inputs['city_id_x']), training=training))
        embedded.append(self.device_emb_dropout(
            self.device_embeddings(inputs['device_class']), training=training))
        embedded.append(self.affiliate_emb_dropout(
            self.affiliate_embeddings(inputs['affiliate_id']), training=training))
        embedded.append(self.hotel_country_emb_dropout(
            self.hotel_country_embeddings(inputs['hotel_country']), training=training))
        most_probable_target_cities = [tf.squeeze(self.target_city_embeddings(inputs[key]))
                                       for key in self.most_probable_cities_keys]  # (batch_size, emb_dim_tc)
        booker_country = tf.squeeze(self.booker_country_embeddings(
            inputs['booker_country']))  # (batch_size, emb_dim_bc)

        # Add checkin/checkout information
        embedded.extend([tf.expand_dims(inputs['checkin_sin'], axis=-1),  # list of (batch_size, max_seq_len, 1)
                         tf.expand_dims(inputs['checkin_cos'], axis=-1),
                         tf.expand_dims(inputs['checkout_sin'], axis=-1),
                         tf.expand_dims(inputs['checkout_cos'], axis=-1)])

        # Concatenated time features
        x = tf.concat(embedded, axis=-1, name='embedded_inputs')  # (batch_size, max_seq_len, sum(emb_dims))

        # Project to a unified space
        x = self.pre_lstm_projection(x)  # (batch_size, max_seq_len, pre_lstm_projection_units)

        # LSTM
        x = self.lstm(x, mask=mask)  # (batch_size, lstm_units)

        # Concat the non temporal features
        x = tf.concat([x, booker_country] +
                      most_probable_target_cities, axis=1)  # (batch_size, lstm_units+emb_dim_tc+emb_dim_bc)

        # Fully connected
        x = self.fc1(x)  # (batch_size, fc1_units)

        # Batch norm
        x = self.batch_norm1(x)  # (batch_size, fc1_units)

        # Bottleneck
        x = self.bottleneck(x)  # (batch_size, bottleneck_units)

        # Output
        x = self.out(x)  # (batch_size, n_labels)

        return x

    def get_config(self):
        raise DeprecationWarning('Serialization should be updated if needed')


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

            self.top_n_mapping_[source] = [target for target, _ in
                                           transition_counter[source].most_common(n=self.top_n)]
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


# noinspection PyUnusedLocal
def fit_preprocess_pipeline(features: pd.DataFrame, labels: pd.DataFrame, top_n_targets: int = 4, **kwargs) -> Pipeline:
    """
    Create and fit preprocess pipeline

    :param features: Features
    :param labels: Labels
    :param top_n_targets: Top targets to include as features
    :param kwargs: Additional params
    :return: Fitted preprocess pipeline
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
          ColumnTransformer([('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                        unknown_value=-1),
                              ['city_id',
                               'device_class',
                               'affiliate_id',
                               'booker_country',
                               'hotel_country'] + top_n_targets_columns),
                             ('cyclic', CyclicEncoder(), ['checkin',
                                                          'checkout'])],
                            remainder='passthrough')),
         ('unknown2zero',  # Step 4 - Convert unknowns from -1 to 0, by adding 1
          ColumnTransformer([('add_1',
                              FunctionTransformer(lambda x: x + 1), [0,  # City ID
                                                                     2,  # Affiliate
                                                                     4])],  # Hotel country
                            remainder='passthrough')),
         ('to_df',
          FunctionTransformer(lambda x: pd.DataFrame(x, columns=['city_id',
                                                                 'affiliate_id',
                                                                 'hotel_country',
                                                                 'device_class',
                                                                 'booker_country'] + top_n_targets_columns + [
                                                                    'checkin_sin', 'checkin_cos',
                                                                    'checkout_sin', 'checkout_cos',
                                                                    'utrip_id'])))])

    preprocess.fit(features, labels)

    return preprocess


def processed_to_dataset(features: pd.DataFrame,
                         labels: pd.DataFrame,
                         batch_size: int,
                         pre_shuffle: bool = False) -> tf.data.Dataset:
    """
    Convert the preprocessed data to `tf.data.Dataset` object

    :param features: Processed features
    :param labels: Labels
    :param batch_size: Batch size
    :param pre_shuffle: Shuffle utrip IDs before yielding groups
    :return: Dataset as tf.data.Dataset
    """

    # Merge
    merged = pd.merge(left=features, right=labels, on='utrip_id')

    # Get the keys of the most probable cities
    most_probable_cities_keys = [k for k in merged.columns if k.endswith('most_probable_city_id')]

    # Samples generator
    def gen():
        # Group to samples
        samples = [g.to_dict(orient='list') for _, g in merged.groupby('utrip_id', sort=False)]

        if pre_shuffle:  # Shuffle, should be used only for training
            shuffle(samples)

        for sample in samples:
            # Remove utrip_id
            sample.pop('utrip_id')

            # separate label
            y = sample.pop('city_id_y')

            # Flatten values that are not temporal (consistent over time)
            sample_features = dict()
            for k, v in sample.items():
                # Flat values
                if k.endswith('most_probable_city_id') or k == 'booker_country':
                    sample_features[k] = np.array(v[-1])
                else:
                    sample_features[k] = np.array(v)

            yield sample_features, y[-1]

    # Construct output signature of the features
    output_signature_x = {'city_id_x': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          'affiliate_id': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          'hotel_country': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          'device_class': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                          'booker_country': tf.TensorSpec(shape=(), dtype=tf.int32)}
    output_signature_x.update({k: tf.TensorSpec(shape=(), dtype=tf.int32) for k in most_probable_cities_keys})
    output_signature_x.update({'checkin_sin': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'checkin_cos': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'checkout_sin': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'checkout_cos': tf.TensorSpec(shape=(None,), dtype=tf.float32)})

    # Create dataset object
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_signature=(output_signature_x, tf.TensorSpec(shape=(), dtype=tf.int32)))

    # Prepare padding scheme
    padded_shapes = {
        'city_id_x': [None],
        'device_class': [None],
        'affiliate_id': [None],
        'booker_country': [],  # No padding
        'hotel_country': [None]}
    padding_values = {
        'city_id_x': 0,
        'device_class': 0,
        'affiliate_id': 0,
        'booker_country': 0,  # No padding
        'hotel_country': 0}
    padded_shapes.update({k: [] for k in most_probable_cities_keys})
    padding_values.update({k: 0 for k in most_probable_cities_keys})
    padded_shapes.update({'checkin_sin': [None],
                          'checkin_cos': [None],
                          'checkout_sin': [None],
                          'checkout_cos': [None]})
    padding_values.update({'checkin_sin': 0.0,
                           'checkin_cos': 0.0,
                           'checkout_sin': 0.0,
                           'checkout_cos': 0.0})

    # Batch and pad
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padding_values=(padding_values, 0),
                                   padded_shapes=(padded_shapes, [])).prefetch(1)
    return dataset


def get_model(**model_init_kwargs) -> tf.keras.Model:
    """
    Get compiled version of model

    :return: Compiled model
    """

    # Init
    model = VanillaLSTM(**model_init_kwargs)

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=4)])

    return model
