from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


# noinspection PyPep8Naming
class BinaryEncoder(TransformerMixin, BaseEstimator):
    """
    Converts number to binary format. Accepts only integers, greater than 0.

    Examples
    --------

    .. code-block:: python3

        from models.transformers import BinaryEncoder

        # Create encoder
        be = BinaryEncoder()

        # Data
        X =  np.array([1, 0, 4])

        print(be.fit_transform(X))
        [[0 0 1]
         [0 0 0]
         [1 0 0]]
    """

    def __init__(self, maximum: Optional[List[int]] = None):
        """

        :param maximum: Maximum values per column
        """

        self.maximum = maximum

    # noinspection PyUnusedLocal,PyAttributeOutsideInit
    def fit(self, X, y=None):

        # Validate input
        check_array(X, dtype=None)
        data = X.values if isinstance(X, pd.DataFrame) else X

        # Figure out the number of columns needed to represent all the numbers
        if not self.maximum:
            self.maximum = np.max(data, axis=0).astype(int)

        self.depth_ = list(map(self._get_depth, self.maximum))

        # Get relevant indices for trimming all zero columns
        self.relevant_indices_ = [slice(-d, None) for d in self.depth_]

        return self

    def transform(self, X):

        # Input validation
        check_array(X, dtype=None)
        data = X.values.copy() if isinstance(X, pd.DataFrame) else X.copy()

        data = data.astype(int)

        # Input validation
        if np.any(np.min(data) < 0):
            raise ValueError('BinaryEncoder only accepts positive values')

        if np.any([self._get_depth((value := max_x)) >
                   boundary for max_x, boundary in zip(np.max(data, axis=0), self.depth_)]):
            raise ValueError(f"Value too large, can't binarize ({value=})")

        def to_bits(array: np.ndarray, indices: slice) -> np.ndarray:
            """ Convert a one dimensional array to its bits representation resulting in a matrix """

            # Convert to bits
            array = np.reshape(array, newshape=(-1, 1)).astype(">i2")
            array = np.unpackbits(array.view(np.uint8), axis=1)

            # Remove redundant columns
            array = array[:, indices]

            return array

        # Convert to bits
        transformed = np.concatenate([to_bits(data[:, i], indices) for i, indices in enumerate(self.relevant_indices_)],
                                     axis=-1)

        return transformed

    @staticmethod
    def _get_depth(number: int) -> int:
        """
        Get the number of digits necessary to represent number as binary

        :param number: Number
        :return: Number of digits necessary to represent number as binary
        """

        return len(bin(np.max(number))[2:])
