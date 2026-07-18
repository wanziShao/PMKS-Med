from .base_language_model import BaseLanguageModel
from .model_adapter import *

registed_language_models = {
    'qwen': Qwen,
}

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    raise ValueError(f"No registered model found for name {model_name}")
