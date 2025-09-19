
import torch
import pytest

from pathlib import Path

from fgrinmet.utils.coordinates import *
from fgrinmet.utils.io_data import *
from fgrinmet.globvar import DEVICE
from config import LOCAL_DATA_DIR

@pytest.mark.parametrize("w_params", [
    {"7": 1},
    {"1": 1, "2": 1, "3": 1, "4": 1},
])
def test_rw_json(w_params):
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = get_file_route(LOCAL_DATA_DIR, "OI_test.json")
    write_json(w_params, file_path)
    r_params =  read_json(file_path)
    Path(file_path).unlink()

    assert w_params == r_params

@pytest.mark.parametrize("shape, pix_size", [
    ((4,), 1.0),
    ((4,), 0.5),
    ((4, 4), 1.0),
    ((4, 6), (1.0, 2.0)),
])
def test_coord_pytorch_shape_and_dtype(shape, pix_size):
    Xi = coord_pytorch(shape, pix_size, dtype=torch.float32, device=DEVICE)

    assert isinstance(Xi, tuple)
    assert len(Xi) == len(shape)

    for i, x in enumerate(Xi):
        assert isinstance(x, torch.Tensor)
        assert x.shape == tuple(shape)
        assert x.dtype == torch.float32
        assert x.device.type == DEVICE.type

        mid_val = x[tuple(s // 2 for s in shape)]
        assert torch.isclose(mid_val, torch.tensor(0.0, dtype=torch.float32, device=x.device))


@pytest.mark.parametrize("shape, pix_size", [
    ((4,), 1.0),
    ((4,), 0.5),
    ((4, 4), 1.0),
    ((4, 6), (1.0, 2.0)),
    ((4, 6, 9), (1.0, 2.0, 3.0))
])
def test_fft_coord_pytorch_matches_fftfreq(shape, pix_size):
    Fi = fft_coord_pytorch(shape, pix_size, dtype=torch.float64, device=DEVICE)

    assert isinstance(Fi, tuple)
    assert len(Fi) == len(shape)

    for i, f in enumerate(Fi):
        assert isinstance(f, torch.Tensor)
        assert f.shape == tuple(shape)
        assert f.dtype == torch.float64
        assert f.device.type == DEVICE.type

        expected = torch.fft.fftshift(
            torch.fft.fftfreq(shape[i], d=(pix_size[i] if isinstance(pix_size, tuple) else pix_size),
                              dtype=torch.float64)
        ).to(f.device)

        unique_vals = f.clone()
        ex_dims = [j for j in range(len(Fi)) if j!=i]
        for j in ex_dims:
            unique_vals = torch.unique(unique_vals, dim=j)
        assert torch.allclose(torch.sort(unique_vals.flatten())[0], torch.sort(expected)[0])

@pytest.mark.parametrize("shape, pix_size", [
    ((8,), 0.5)
])
def test_coord_and_fft_consistency(shape, pix_size):
    Xi = coord_pytorch(shape, pix_size, dtype=torch.float64, device=DEVICE)
    Fi = fft_coord_pytorch(shape, pix_size, dtype=torch.float64, device=DEVICE)

    x = Xi[0][:,]
    f = Fi[0][:,]

    nyquist = 1 / (2 * pix_size)
    assert torch.isclose(torch.max(torch.abs(f)),
                         torch.tensor(nyquist, dtype=f.dtype, device=f.device))