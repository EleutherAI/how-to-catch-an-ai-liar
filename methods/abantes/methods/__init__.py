"""__init__ — internal module.
"""
import importlib


def load(name: str = "pm01"):
    mod = importlib.import_module(f"methods.{name}")
    assert hasattr(mod, "build"), f"method '{name}' must expose build(model_id, lora_id)"
    if not hasattr(mod, "NAME"):
        mod.NAME = name
    return mod
