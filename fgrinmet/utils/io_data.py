import json
from typing import Any
from pathlib import Path

def write_json(data: Any, route: str, **kwargs):    
    
    with open(route, "w") as outfile:
        json.dump(data, outfile, **kwargs)

def read_json(route: str) -> Any:

    with open(route, "r") as outfile:
        data = json.load(outfile)
    return data

def get_file_route(directory: str | Path,
                   name: str = "new_file.txt") -> str:
    
    if isinstance(directory, str):
        route = Path(directory) / name
    elif isinstance(directory, Path):
        route = directory / name
    else:
        raise(TypeError("Incompatible data type as directory, must be string or Path."))
     
    return str(route)