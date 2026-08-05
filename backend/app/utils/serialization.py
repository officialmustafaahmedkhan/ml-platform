"""JSON (de)serialization helpers used across services/routes."""
import json
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """json.JSONEncoder that knows how to serialize numpy scalars/arrays."""

    def default(self, o: Any) -> Any:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types inside common containers to native JSON."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), cls=NumpyEncoder, default=str)


def loads(raw: str | None, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default
