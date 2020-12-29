from typing import List, Optional

import numpy as np
import pytest

from models.transformers import BinaryEncoder


@pytest.mark.parametrize(
    "inputs,outputs",
    [(np.array([1, 0, 5]).reshape(-1, 1), np.array([[0, 0, 1],
                                                    [0, 0, 0],
                                                    [1, 0, 1]])),
     (np.array([30_000, 0, 5]).reshape(-1, 1), np.array([[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]]))],
)
def test_encoder_output_no_encoding(inputs: np.ndarray, outputs: np.ndarray):
    """
    Check that binary encoder encodes as expected different types of input
    """

    binary_encoder = BinaryEncoder()

    transformed = binary_encoder.fit_transform(inputs)

    assert np.all(transformed == outputs)


@pytest.mark.parametrize(
    "inputs,outputs,encode",
    [(np.array([[1, 'no'],
                [0, 'yes'],
                [5, 'yes']]),
      np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 1]]),
      [False, True]),
     (np.array([[25, 'no'],
                [30, 'yes'],
                [55, 'yes']]),
      np.array([[0, 0, 0],
                [0, 1, 1],
                [1, 0, 1]]),
      [True, True])])
def test_encoder_output_with_encoding(inputs: np.ndarray,
                                      outputs: np.ndarray,
                                      encode: Optional[List[bool]]):
    """
    Check that binary encoder encodes as expected different types of input, using internal encoding
    """

    binary_encoder = BinaryEncoder(pre_encode=encode)

    transformed = binary_encoder.fit_transform(inputs)

    assert np.all(transformed == outputs)


def test_encoder_output_with_init_max():
    """
    Check that binary encoder encodes as expected when given maximum in constructor
    """

    binary_encoder = BinaryEncoder([30_000])
    inputs = np.array([30_001, 0, 5]).reshape(-1, 1)
    outputs = np.array([[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])

    transformed = binary_encoder.fit_transform(inputs)

    assert np.all(transformed == outputs)
