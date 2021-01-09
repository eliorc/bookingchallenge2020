from typing import Callable, Tuple

from models import trans_prob
from models import vanilla_lstm

_MODELS = {'trans_prob': trans_prob.fit_pipeline,
           'benchmark': trans_prob.fit_pipeline,
           'vanilla_lstm': (vanilla_lstm.fit_preprocess_pipeline,
                            vanilla_lstm.processed_to_dataset,
                            vanilla_lstm.get_model)}

AVAILABLE_MODELS = set(_MODELS.keys())


def get_fitting_method(name: str) -> Tuple[Callable, Callable, Callable]:
    """
    Load model Model/pipelines by name

    :param name: Model/pipelines name
    :return: fitted pipeline
    """

    # Deprecate benchmark model, due to the the change of the returned values by this function
    if name in {'trans_prob', 'benchmark'}:
        raise DeprecationWarning('Benchmark model can\'t be used with the new loading method')

    return _MODELS[name]
