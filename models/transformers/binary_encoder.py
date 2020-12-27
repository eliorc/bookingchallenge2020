from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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

    def __init__(self, maximum: Optional[int] = None):

        self.maximum = maximum

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):

        # Figure out the number of columns needed to represent all the numbers
        # noinspection PyAttributeOutsideInit
        self.depth_ = self._get_depth(self.maximum or np.max(X))  # Ignore np.max(X) if supplied in init

        # Get relevant indices for trimming all zero columns
        # noinspection PyAttributeOutsideInit
        self._relevant_indices = slice(-self.depth_, None)

        return self

    def transform(self, X):

        # Input validation
        if np.min(X) < 0:
            raise ValueError('BinaryEncoder only accepts positive values')
        if self._get_depth((value := np.max(X))) > self.depth_:
            raise ValueError(f"Value too large, can't binarize ({value=})")

        # Convert to bits
        X = np.reshape(X, newshape=(-1, 1)).astype(">i2")
        X = np.unpackbits(X.view(np.uint8), axis=1)

        # Remove redundant columns
        X = X[:, self._relevant_indices]

        return X

    @staticmethod
    def _get_depth(number: int) -> int:
        """
        Get the number of digits necessary to represent number as binary

        :param number: Number
        :return: Number of digits necessary to represent number as binary
        """

        return len(bin(np.max(number))[2:])
