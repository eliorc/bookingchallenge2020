import warnings
from pathlib import Path
from typing import Tuple

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
