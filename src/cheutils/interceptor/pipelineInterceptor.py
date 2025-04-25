import pandas as pd
from abc import ABC, abstractmethod

class PipelineInterceptor(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'apply') and
                callable(subclass.apply) or
                NotImplemented)

    @abstractmethod
    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> pd.DataFrame:
        raise NotImplementedError