from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array


class CyclicEncoder(TransformerMixin, BaseEstimator):
    """
    Encode numerical features in a cyclic manner.
    Splits each variable into two features which represent sine and cosine
    """

    def __init__(self, maximum: Optional[List[int]] = None):
        self.maximum = maximum

    # noinspection PyUnusedLocal
    def fit(self, features, labels=None):
        self.maximum = np.array(self.maximum) if self.maximum is not None else None
        return self

    def transform(self, features):
        # Validate input
        check_array(features)

        data = features.values.copy() if isinstance(features, pd.DataFrame) else features.copy()

        # Take numpy data if given as dataframe
        data = data.values if isinstance(data, pd.DataFrame) else data
        data = data.astype(float)

        # Infer maximum if not given
        if self.maximum is None:
            self.maximum = np.max(data, axis=0)

        # Create as cosine/sine features
        norm = (2 * np.pi * data) / self.maximum
        sin = np.sin(norm)
        cos = np.cos(norm)

        # Interleave to keep order
        transformed = np.empty((norm.shape[0], norm.shape[1] * 2), dtype=norm.dtype)
        transformed[:, 0::2] = sin
        transformed[:, 1::2] = cos

        return transformed
