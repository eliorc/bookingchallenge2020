import warnings

import conf

import pandas as pd


def load_raw_data() -> pd.DataFrame:
    """
    Load training set as given by booking

    :return: Raw train set
    """

    # Avoid redundant warning - https://stackoverflow.com/a/46721064/5368083
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        return pd.read_csv(conf.DATA_DIR / 'raw/booking_train_set.csv', dtype={'user_id': 'int32',
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
