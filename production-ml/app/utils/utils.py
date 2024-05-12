from typing import Any, Dict

from pydantic import BaseModel

from app.types.production_history_model import FactoryProductionHistory
from app.utils.common import to_camel


def convert_keys_to_strings(data: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in data.items()}

def convert_string_keys_to_floats(data: Dict[str, Any]) -> Dict[float, Any]:
    return {float(k): v for k, v in data.items()}

def to_snake(string: str) -> str:
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def serialize_production_history(history: FactoryProductionHistory):
    history_data = history.model_dump(by_alias=True)

    # Convert all camelCase keys to snake_case for MongoDB storage and internal use
    def recursive_snake_case(d):
        if isinstance(d, dict):
            return {to_snake(k): recursive_snake_case(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_snake_case(item) for item in d]
        else:
            return d

    snake_case_data = recursive_snake_case(history_data)
    return snake_case_data


def deserialize_to_model(data: dict, model: BaseModel):
    # Convert all snake_case keys to camelCase as per model alias definitions
    def recursive_camel_case(d):
        if isinstance(d, dict):
            return {to_camel(str(k)): recursive_camel_case(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_camel_case(v) for v in d]
        else:
            return d

    camel_case_data = recursive_camel_case(data)
    return model(**camel_case_data)