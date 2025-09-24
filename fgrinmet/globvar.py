import torch

# Automatically detect device
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps")
          if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else torch.device("cpu"))