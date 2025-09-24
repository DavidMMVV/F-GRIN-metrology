import pandas as pd # type: ignore

from pathlib import Path

from torch.utils.data import Dataset
import torch

class DatasetField(Dataset):
    """Dataset for the input and output fields"""
    def __init__(self, datadir: str):
        """For initializing the dataset we need a directory with 
        input fields under the name of "input" and a directory with
        output fields named as "output" and a csv file with the extra
        data corresponding to each pair of input and output fields named "data".

        Args:
            datadir (str): Directory 
        """

        self.data = pd.read_csv(Path(datadir) / "data.csv")
        self.input_dir = Path(datadir) / "input"
        self.output_dir = Path(datadir) / "output"

    def __len__(self) -> int:
        return self.data.shape[0]
    def __getitem__(self, index: int):
        rindex, wavelength, in_field_num, out_field_num = self.data.iloc[index]
        in_field = read_comp_image(self.input_dir / in_field_num)

        return super().__getitem__(index)

def read_comp_image(path: str) -> torch.Tensor:
    

if __name__ == "__main__":
    import numpy as np
    d = np.arange(100).reshape(-1,2)
    df = pd.DataFrame(d, columns=["a", "b"])
    k = df.iloc(0)
    print(k)
    a, b = df.iloc[0]
    print(a,b)