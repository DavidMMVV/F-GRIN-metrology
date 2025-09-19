from fgrinmet.globvar import BASE_DIR, DATA_DIR, LOCAL_DATA_DIR

def test_base_dir():
    assert BASE_DIR.exists()
def test_data_dir():
    assert DATA_DIR.exists()
def test_local_data_dir():
    assert LOCAL_DATA_DIR.exists()