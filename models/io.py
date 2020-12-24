from typing import Callable

from models import trans_prob

_MODELS = {'trans_prob': trans_prob.fit_pipeline,
           'benchmark': trans_prob.fit_pipeline}
AVAILABLE_MODELS = set(_MODELS.keys())


def get_fitting_method(name: str) -> Callable:
    """
    Load model Model/pipeline by name

    :param name: Model/pipeline name
    :return: fitted pipeline
    """

    return _MODELS[name]
