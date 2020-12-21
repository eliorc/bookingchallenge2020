from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def calculate_transition_probabilities(size: int,
                                       source: np.ndarray,
                                       target: np.ndarray,
                                       fill_untraveled_with_prior: bool = False) -> np.ndarray:
    """
    Calculate transition probabilities between sources and targets

    :param size: Size of each dimension of the transition probability matrix
    :param source: Sources - should be encoded (numbers from 0 onwards)
    :param target: Targets - should be encoded (numbers from 0 onwards)
    :param fill_untraveled_with_prior: Fill sources which never been traveled from with prior of targets' travels
    :return: Transition probability matrix
    """

    # Prerequisites
    target_prior = None

    # Init transition matrix
    transition_matrix = np.zeros(shape=(size, size), dtype=float)

    # Fill matrix
    try:
        for source_, target_ in zip(source.flatten(), target.flatten()):
            transition_matrix[source_, target_] += 1
    except IndexError:
        raise ValueError('Sources and targets must be encoded from 0 onwards.')

    if fill_untraveled_with_prior:
        target_prior = np.sum(transition_matrix, axis=0)
        target_prior /= np.sum(target_prior)

    # Normalize
    n_transitions = transition_matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by 0
        transition_matrix /= np.reshape(n_transitions, newshape=(-1, 1))

    transition_matrix = np.nan_to_num(transition_matrix)

    if fill_untraveled_with_prior:
        transition_matrix[~np.sum(transition_matrix, axis=1).astype(bool), :] = target_prior

    return transition_matrix


# noinspection PyAttributeOutsideInit
class TransProb(BaseEstimator, ClassifierMixin):
    """
    Transition probability model.
    Predicts the next city probability using the prior calculated from training

    Arguments
    ---------

     - `X` (``np.ndarray``): Source city IDs - should be encoded (numbers from 0 onwards)
     - `y` (``np.ndarray``): Destination City IDs - should be encoded (numbers from 0 onwards)
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TransProb':
        """

        :param X: Source city IDs - should be encoded (numbers from 0 onwards)
        :param y: Destination City IDs - should be encoded (numbers from 0 onwards)
        :return: self
        """

        # Check that X and y have correct shape
        # noinspection PyPep8Naming
        X, y = check_X_y(X, y, ensure_2d=False)

        # Calculate transition matrix
        # * Transition matrix will hold the transition probability between each source to target.
        # * In case where the source has never been traveled from (all zero row) it will be assigned the prior
        # * of the target
        self._transition_matrix = calculate_transition_probabilities(size=len(set(X.flatten()).union(y.flatten())),
                                                                     source=X,
                                                                     target=y,
                                                                     fill_untraveled_with_prior=True)

        # Calculate prediction table
        # noinspection PyAttributeOutsideInit
        self.prediction_table_ = np.argsort(-self._transition_matrix, axis=1)

        return self

    def predict(self, X) -> np.ndarray:
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        # noinspection PyPep8Naming
        X = check_array(X, ensure_2d=False)

        try:
            return self.prediction_table_[X, :]
        except IndexError:
            raise ValueError('X must be encoded from 0 onwards.')

    def predict_proba(self, X) -> np.ndarray:
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        # noinspection PyPep8Naming
        X = check_array(X, ensure_2d=False)

        try:
            return self._transition_matrix[X, :]
        except IndexError:
            raise ValueError('X must be encoded from 0 onwards.')
