from importlib import import_module

__all__ = ["Hexels3DOptions", "Hexels3DRenderer", "Hexels3DTrainer"]

_MODULE_BY_NAME = {
    "Hexels3DOptions": ".options",
    "Hexels3DRenderer": ".renderer",
    "Hexels3DTrainer": ".trainer",
}


def __getattr__(name: str) -> object:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULE_BY_NAME[name], __name__), name)
    globals()[name] = value
    return value
