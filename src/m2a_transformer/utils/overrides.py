"""Apply CLI overrides of the form `section.key=value` to an AppConfig.

This is the lightweight equivalent of Hydra's override syntax — enough to
do quick sweeps without pulling in a heavy dep. Examples::

    --set model.d_model=768 --set training.learning_rate=1e-4

Values are parsed with `yaml.safe_load` so they get the right Python type
automatically (`true`, `0.001`, `[1,2,4]`, …).
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterable

import yaml

from m2a_transformer.config import AppConfig
from m2a_transformer.utils.logger import logger


def _coerce(value: Any, target_type) -> Any:
    """Make `value` match `target_type` when reasonable.

    YAML 1.1's `safe_load` does NOT parse scientific notation without a dot
    (`1e-4` stays a string), and CLI users naturally type it. We coerce
    string → float / int / bool when the target field is typed that way."""
    if target_type is float and isinstance(value, str):
        return float(value)
    if target_type is int and isinstance(value, str):
        return int(value)
    if target_type is bool and isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y", "on")
    return value


def apply_overrides(config: AppConfig, overrides: Iterable[str]) -> AppConfig:
    """Mutate `config` in place from a list of `section.key=value` strings."""
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Bad --set spec '{raw}'. Expected 'section.key=value'.")
        path, value = raw.split("=", 1)
        keys = [k.strip() for k in path.strip().split(".") if k.strip()]
        if not keys:
            raise ValueError(f"Empty key in --set '{raw}'")

        parsed = yaml.safe_load(value)
        # Walk to the parent dataclass.
        target = config
        for k in keys[:-1]:
            if not hasattr(target, k):
                raise KeyError(
                    f"--set '{raw}': unknown section '{k}' on {type(target).__name__}"
                )
            target = getattr(target, k)
            if not is_dataclass(target):
                raise TypeError(
                    f"--set '{raw}': '{k}' is not a config section."
                )
        final = keys[-1]
        if not hasattr(target, final):
            raise KeyError(
                f"--set '{raw}': unknown field '{final}' on {type(target).__name__}"
            )
        # Coerce to the field's declared type when possible. With
        # `from __future__ import annotations` enabled, `f.type` is a string
        # like "float", not the actual class — match by name.
        target_type_str = next(
            (str(f.type) for f in fields(target) if f.name == final), ""
        )
        type_map = {"float": float, "int": int, "bool": bool}
        if target_type_str in type_map:
            parsed = _coerce(parsed, type_map[target_type_str])
        setattr(target, final, parsed)
        logger.info(f"override: {path} = {parsed!r}")
    return config
