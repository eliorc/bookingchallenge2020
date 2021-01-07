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

    def __init__(self, minimum: Optional[List[int]] = None, maximum: Optional[List[int]] = None):
        self.minimum = minimum
        self.maximum = maximum

    # noinspection PyUnusedLocal
    def fit(self, features, labels=None):
        # Convert to numpy arrays
        self.maximum = np.array(self.maximum if self.maximum is not None else np.max(features, axis=0))
        self.minimum = np.array(self.minimum if self.minimum is not None else np.min(features, axis=0))
        return self

    def transform(self, features):
        # Validate input
        check_array(features)

        data = features.values.copy() if isinstance(features, pd.DataFrame) else features.copy()

        # Take numpy data if given as dataframe
        data = data.values if isinstance(data, pd.DataFrame) else data
        data = data.astype(float)

        # Create as cosine/sine features
        # noinspection PyUnresolvedReferences
        norm = (2 * np.pi * (data - self.minimum)) / (self.maximum - self.minimum)
        sin = np.sin(norm)
        cos = np.cos(norm)

        # Interleave to keep order
        transformed = np.empty((norm.shape[0], norm.shape[1] * 2), dtype=norm.dtype)
        transformed[:, 0::2] = sin
        transformed[:, 1::2] = cos

        return transformed
