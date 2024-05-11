from typing import Any, Dict


def convert_float_keys_to_strings(data: Dict[float, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in data.items()}

def convert_string_keys_to_floats(data: Dict[str, Any]) -> Dict[float, Any]:
    return {float(k): v for k, v in data.items()}
