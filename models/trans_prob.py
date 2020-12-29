from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_is_fitted


def calculate_transition_probabilities(size: int,
                                       source: np.ndarray,
                                       target: np.ndarray,
                                       return_destination_prior: bool = False) -> Tuple[np.ndarray,
                                                                                        Optional[np.ndarray]]:
    """
    Calculate transition probabilities between sources and targets

    :param size: Size of each dimension of the transition probability matrix
    :param source: Sources - should be encoded (numbers from 0 onwards)
    :param target: Targets - should be encoded (numbers from 0 onwards)
    :param return_destination_prior: Return prior probability of destination travel
    :return: Transition probability matrix, and optionally destination prior vector
    """

    # Prerequisites
    target_prior = None

    # Init transition matrix
    transition_matrix = np.zeros(shape=(size, size), dtype=float)

    # Fill matrix with counts
    try:
        for source_, target_ in zip(source.flatten(), target.flatten()):
            transition_matrix[source_, target_] += 1
    except IndexError:
        raise ValueError('Sources and targets must be encoded from 0 onwards.')

    if return_destination_prior:
        target_prior = np.sum(transition_matrix, axis=0)
        target_prior /= np.sum(target_prior)

    # Normalize
    n_transitions = transition_matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by 0
        transition_matrix /= np.reshape(n_transitions, newshape=(-1, 1))
    transition_matrix = np.nan_to_num(transition_matrix)

    return transition_matrix, target_prior


# noinspection PyAttributeOutsideInit,PyPep8Naming
class TransProb(BaseEstimator, ClassifierMixin):
    """
    Transition probability model.
    Predicts the next city probability using the prior calculated from training

    Arguments
    ---------

     - `X` (``np.ndarray``): Source city IDs - should be encoded (numbers from 0 onwards)
     - `y` (``np.ndarray``): Destination City IDs - should be encoded (numbers from 0 onwards)
    """

    def __init__(self, n_cities: int, top_n: int = 1):
        """

        :param n_cities: Number of cities in the data
        :param top_n: Top N to predict by default
        """

        self.top_n = top_n
        self.n_cities = n_cities

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
        # * Zero elements in the matrix will be filled using the prior of targets in a manner where the filled elements
        # * are all smaller than the true probabilities. This will be done by giving them negative values
        # * 1-(prior probability) resulting in a matrix where in each row the highest elements are decided by the
        # * probability of truly travelled destinations and then smaller elements representing the prior probability
        # * of target travel.
        self._transition_matrix, target_prior = calculate_transition_probabilities(size=self.n_cities,
                                                                                   source=X,
                                                                                   target=y,
                                                                                   return_destination_prior=True)
        # Negative values, but keep order for sorting
        target_prior = -(1 - target_prior)

        # Swap all zero elements with the adjusted target prior
        self._transition_matrix = np.where(self._transition_matrix == 0, target_prior,
                                           self._transition_matrix).astype(np.float32)

        # Calculate prediction table, element[i, j] is the jth moth probable city to travel to from i
        # noinspection PyAttributeOutsideInit
        self.prediction_table_ = np.argsort(-self._transition_matrix, axis=1).astype(np.int32)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check is fit had been called
        check_is_fitted(self)

        X = X.flatten()

        try:
            if self.top_n == 1:
                predicted_labels = self.prediction_table_[X, 0]
            else:
                predicted_labels = self.prediction_table_[X, :self.top_n]
        except IndexError:
            raise ValueError('X must be encoded from 0 onwards.')

        # One hot
        if self.top_n == 1:  # Predicts top 1
            return tf.one_hot(predicted_labels, depth=self.prediction_table_.shape[1]).numpy()
        else:  # Predicts top N > 1
            return np.max(  # np.max will make the one_hot able to return multi label output
                tf.one_hot(predicted_labels, depth=self.prediction_table_.shape[1]).numpy(),
                axis=1)

    def predict_proba(self, X) -> np.ndarray:
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = X.flatten()

        try:
            return self._transition_matrix[X, :]
        except IndexError:
            raise ValueError('X must be encoded from 0 onwards.')


class ModelingTable(TransformerMixin, BaseEstimator):
    """
    Takes in raw data and discards of irrelevant features

    Transforms into numpy array with only the relevant features
    """

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        return self

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def transform(self, X):
        # X already excludes the last city of each trip

        # Extract only relevant rows of city_id - the last of each trip
        X = X.groupby('utrip_id').tail(1)['city_id'].values

        return X


# noinspection PyUnusedLocal
def fit_pipeline(features: pd.DataFrame, labels: pd.DataFrame, **kwargs) -> Pipeline:
    """
    Create and fit pipeline

    :param features: Features, should include the entire raw data format, cities already encoded
    :param labels: Labels, should include only 'utrip_id' and 'city_id' columns with already encoded cities
    :param kwargs: Support additional arguments for fitting
    :return: Fitted pipeline
    """

    # Build pipeline
    n_cities = features['city_id'].nunique() + 1  # +1 for unknown city
    pipeline = Pipeline([('modeling_table', ModelingTable()),
                         ('trans_prob', TransProb(n_cities=n_cities))])
    # Fit
    pipeline.fit(features, labels['city_id'].values)

    return pipeline
