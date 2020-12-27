import numpy as np
import pytest

from models.transformers import BinaryEncoder


@pytest.mark.parametrize(
    "inputs,outputs",
    [(np.array([1, 0, 5]), np.array([[0, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 1]])),
     (np.array([30_000, 0, 5]), np.array([[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]]))],
)
def test_encoder_output(inputs: np.ndarray, outputs: np.ndarray):
    """
    Check that binary encoder encodes as expected different types of input
    """

    binary_encoder = BinaryEncoder()

    transformed = binary_encoder.fit_transform(inputs)

    assert np.all(transformed == outputs)
