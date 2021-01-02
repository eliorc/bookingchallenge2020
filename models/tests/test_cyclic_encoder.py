import numpy as np
import pytest

from models.transformers import CyclicEncoder


@pytest.mark.parametrize(
    "inputs,",
    [np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1),
     np.array([2, 3, 4, 5, 6, 7]).reshape(-1, 1)],
)
def test_encoder_output(inputs: np.ndarray):
    """
    Check that cyclic encoder encodes as expected different types of input
    """

    cyclic_encoder = CyclicEncoder()

    cyclic_encoder.fit(inputs)

    assert np.all(np.isclose(cyclic_encoder.transform(inputs[[0, 1], :]),
                             cyclic_encoder.transform(inputs[[-2, -1], :] + 1)))  # Cycle plus one step
