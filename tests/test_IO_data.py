from src.essentials.representation import *
from src.essentials.IO_data import *
from pathlib import Path
from config import LOCAL_DATA_DIR

def test_rw_json():
    w_params = {"7": 1} 
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = get_file_route(LOCAL_DATA_DIR, "OI_test.json")
    write_json(w_params, file_path)
    r_params =  read_json(file_path)
    Path(file_path).unlink()

    assert w_params == r_params
    

