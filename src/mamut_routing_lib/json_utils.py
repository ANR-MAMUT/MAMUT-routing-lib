from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Any


def custom_json_encode(
    obj: Any,
    indent: int = 4,
    level: int = 0,
    sort_keys: bool = False,
) -> str:
    current_indent = " " * (indent * level)
    next_indent = " " * (indent * (level + 1))

    if isinstance(obj, enum.Enum):
        return json.dumps(obj.value)

    if isinstance(obj, dict):
        items_iter = obj.items()
        if sort_keys:
            items_iter = sorted(items_iter, key=lambda kv: kv[0])

        items = []
        for key, value in items_iter:
            key_str = json.dumps(key)
            value_str = custom_json_encode(
                value,
                indent=indent,
                level=level + 1,
                sort_keys=sort_keys,
            )
            items.append(f"{next_indent}{key_str}: {value_str}")
        return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"

    if isinstance(obj, list):
        if not obj:
            return "[]"

        if all(isinstance(x, (int, float)) for x in obj):
            return "[" + ", ".join(json.dumps(x) for x in obj) + "]"

        if all(isinstance(x, list) for x in obj):
            items = [
                next_indent
                + custom_json_encode(
                    inner,
                    indent=indent,
                    level=level + 1,
                    sort_keys=sort_keys,
                )
                for inner in obj
            ]
            return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"

        if all(isinstance(x, tuple) for x in obj):
            items = [next_indent + json.dumps(tup) for tup in obj]
            return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"

        if all(isinstance(x, str) for x in obj):
            return "[" + ", ".join(json.dumps(x) for x in obj) + "]"

        items = [
            next_indent
            + custom_json_encode(
                item,
                indent=indent,
                level=level + 1,
                sort_keys=sort_keys,
            )
            for item in obj
        ]
        return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"

    if isinstance(obj, tuple):
        tuple_items = ", ".join(
            custom_json_encode(
                item,
                indent=indent,
                level=level,
                sort_keys=sort_keys,
            )
            for item in obj
        )
        return f"[{tuple_items}]"

    return json.dumps(obj)


class CustomJsonEncoder(json.JSONEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "indent" not in kwargs:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self._sort_keys = getattr(self, "sort_keys", False)

    def encode(self, obj: Any) -> str:
        indent = self.indent if isinstance(self.indent, int) else 0
        return custom_json_encode(
            obj,
            indent=indent,
            level=0,
            sort_keys=self._sort_keys,
        )


def get_custom_json_string(
    obj: Any,
    indent: int = 4,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        obj,
        cls=CustomJsonEncoder,
        indent=indent,
        sort_keys=sort_keys,
    )


def load_json_from_file(filepath: str | Path) -> Any:
    with Path(filepath).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_to_file(
    obj: Any,
    filepath: str | Path,
    indent: int = 4,
    sort_keys: bool = False,
) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(get_custom_json_string(obj, indent=indent, sort_keys=sort_keys))
        handle.write("\n")
