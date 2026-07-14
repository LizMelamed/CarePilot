from typing import Any, Dict


class SingletonMeta(type):
    """
    A thread-safe-ready, reusable metaclass for creating Singletons.
    """
    # A dictionary to store the single instances of our classes
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Store instance per class
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]