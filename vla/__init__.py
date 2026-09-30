from .memory_vla import MemoryVLA
from .load import available_model_names, available_models, get_model_description, load, load_vla


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
