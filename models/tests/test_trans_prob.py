import numpy as np
import pytest

from models.trans_prob import calculate_transition_probabilities, TransProb


def test_transition_probability_calculation():
    """
    Verify that transition probabilities are calculated correctly
    """

    # Inputs
    sources = np.array([0, 0, 0, 0])
    targets = np.array([0, 1, 2, 3])

    expected_result = np.array([[0.25, 0.25, 0.25, 0.25],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 0.]])

    # Calculate transition matrix
    transition_matrix, target_prior = calculate_transition_probabilities(size=4, source=sources, target=targets,
                                                                         return_destination_prior=True)

    assert np.all(transition_matrix == expected_result)
    assert np.all(target_prior == np.array([[0.25, 0.25, 0.25, 0.25]]))


@pytest.mark.parametrize(
    "sources,targets",
    [(np.array([0, 0, 0, 0]), np.array([0, 35, 0, 0])),  # Should fail because of targets
     (np.array([0, 35, 0, 0]), np.array([0, 0, 0, 0]))],  # Should fail because of sources
)
def test_transition_probability_calculation_unencoded_cities_exception(sources: np.ndarray, targets: np.ndarray):
    """
    Verify that if inputs are not encoded
    """

    # Calculate transition matrix, verify that the exception is being raised
    with pytest.raises(ValueError, match=r"must be encoded from 0 onwards"):
        calculate_transition_probabilities(size=4, source=sources, target=targets)


def test_trans_prob_model_predictions():
    """
    Verify that model predictions are correct
    """

    # Init model
    model = TransProb(n_cities=4)

    # fit
    model.fit(np.array([0, 0, 0, 1]), np.array([1, 1, 2, 3]))

    # Predict
    result = model.predict(np.array([0, 1, 2]))

    # Validate predictions
    assert np.all(result == np.array([[0, 1, 0, 0],
                                      [0, 0, 0, 1],
                                      [0, 1, 0, 0]]))  # Last value is 1 because of prior!

    # Predict top 2
    model.top_n = 2
    result = model.predict(np.array([0, 1, 2]))
    assert np.all(result == np.array([[0, 1, 1, 0],
                                      [0, 1, 0, 1],
                                      [0, 1, 1, 0]]))
