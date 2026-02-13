# save as replace_slashes.py

import json
from typing import Any

# <<< Put your JSON file path here >>>
file_path = r"/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/separate/clinical/DSM_adjusted/DSM_disorders.json"

# Replace '/' in dictionary KEYS too
include_keys = True


def replace_slashes_in_json_file(path: str, include_keys: bool = False) -> None:
    """
    Load a JSON file at `path`, replace every '/' in all string values
    (and optionally in object keys) with 'or', and overwrite the SAME file.
    """
    def _transform(node: Any) -> Any:
        if isinstance(node, str):
            return node.replace("/", " or ")
        if isinstance(node, list):
            return [_transform(item) for item in node]
        if isinstance(node, dict):
            if include_keys:
                return {
                    (_transform(k) if isinstance(k, str) else k): _transform(v)
                    for k, v in node.items()
                }
            else:
                return {k: _transform(v) for k, v in node.items()}
        return node

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = _transform(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    replace_slashes_in_json_file(file_path, include_keys=include_keys)
    print(f"Updated: {file_path} (include_keys={include_keys})")
