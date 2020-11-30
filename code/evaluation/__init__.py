"""
The :mod:`skmultiflow.evaluation` module includes evaluation methods for stream learning.
"""

from .evaluate_prequential import EvaluatePrequential
from .evaluation_data_buffer import EvaluationDataBuffer

__all__ = [
    "EvaluatePrequential",
    "EvaluationDataBuffer"]
