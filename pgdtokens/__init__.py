import torch
from .recover import BERTWordRecover
from .semantic import SemanticHelper

class basicConfig:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)