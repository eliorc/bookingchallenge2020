import random
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import conf


def load_raw_data(data_dir: Path = conf.DATA_DIR, trip_length_threshold: int = 0) -> pd.DataFrame:
    """
    Load training set as given by booking

    :param data_dir: Root data dir
    :param trip_length_threshold: If supplied, will filter out trips that have less destinations than the threshold
    :return: Raw train set
    """

    # Avoid redundant warning - https://stackoverflow.com/a/46721064/5368083
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Load data
        data = pd.read_csv(data_dir / 'raw/booking_train_set.csv', dtype={'user_id': 'int32',
                                                                          # check(in|out) be parsed in parse_dates
                                                                          'checkin': 'str',
                                                                          'checkout': 'str',
                                                                          'city_id': 'int32',
                                                                          'affiliate_id': 'int32',
                                                                          'booker_country': 'str',
                                                                          'hotel_country': 'str',
                                                                          'utrip_id': 'str'},
                           parse_dates=['checkin', 'checkout'],
                           index_col=0)

        # Filter by threshold
        if trip_length_threshold:
            data = data[data.groupby('utrip_id')['user_id'].transform('count') >= trip_length_threshold]

        # Sort by time
        data = data.sort_values(by=['utrip_id', 'checkin'])

        return data


def load_raw_test(data_dir: Path = conf.DATA_DIR) -> pd.DataFrame:
    """
    Load test set as given by booking

    :param data_dir: Root data dir
    :return: Raw test set
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Load data
        data = pd.read_csv(data_dir / 'raw/booking_test_set.csv', dtype={'user_id': 'int32',
                                                                         # check(in|out) be parsed in parse_dates
                                                                         'checkin': 'str',
                                                                         'checkout': 'str',
                                                                         'city_id': 'int32',
                                                                         'affiliate_id': 'int32',
                                                                         'booker_country': 'str',
                                                                         'hotel_country': 'str',
                                                                         'utrip_id': 'str'},
                           parse_dates=['checkin', 'checkout'],
                           index_col=0)

        # Sort by time
        data = data.sort_values(by=['utrip_id', 'checkin'])

    return data


def separate_features_from_label(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given data in raw data format, split into two separate dataframes - features and labels.
    Features will include all the data as in the input data but will remove the last row of each trip.
    Labels will include only the utrip_id and city_id of the last row of each trip.

    :param raw_data: Data in raw format
    :return: Features and labels
    """

    # Add feature for identifying feature and label rows
    raw_data.loc[:, 'is_label'] = raw_data.groupby('utrip_id')['checkin'].transform('max')
    raw_data.loc[:, 'is_label'] = raw_data['checkin'] == raw_data['is_label']

    # Split into features and labels
    label_mask = raw_data['is_label']
    features = raw_data[~label_mask].drop(columns='is_label')
    labels = raw_data[label_mask][['utrip_id', 'city_id']]

    return features, labels


def load_split_train_validation_test(data_dir: Path = conf.DATA_DIR,
                                     validation_frac: float = 0.1,
                                     test_frac: float = 0.1,
                                     trip_length_threshold: int = 0,
                                     seed: int = 1729) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train, validation and test from the raw data. Split is made by utrip IDs

    :param data_dir: Root data dir
    :param validation_frac: Fraction of entire data to assign to validation
    :param test_frac: Fraction of entire data to assign to test
    :param trip_length_threshold: If supplied, will filter out trips that have less destinations than the threshold
    :param seed: Random seed to use, for reproducibility
    :return: Split data
    """

    # Prerequisites
    random.seed(seed)
    np.random.seed(seed)

    # Get raw data
    raw_data = load_raw_data(data_dir=data_dir, trip_length_threshold=trip_length_threshold)

    # Extract trip IDs, to split by
    all_trips = set(raw_data.utrip_id)

    # Figure out sizes for each split
    test_size = int(len(all_trips) * test_frac)
    validation_size = int(len(all_trips) * validation_frac)

    # Assign trips for each split
    test_trips = np.random.choice(list(all_trips), size=test_size, replace=False)
    all_trips -= set(test_trips)
    validation_trips = np.random.choice(list(all_trips), size=validation_size, replace=False)
    train_trips = all_trips - set(validation_trips)

    # Validate split is valid
    assert not train_trips.intersection(validation_trips, test_trips)

    # Validate we have all the trips assigned
    assert set(raw_data.utrip_id) == train_trips.union(test_trips).union(validation_trips)

    return raw_data[raw_data.utrip_id.isin(train_trips)], \
           raw_data[raw_data.utrip_id.isin(validation_trips)], \
           raw_data[raw_data.utrip_id.isin(test_trips)]
