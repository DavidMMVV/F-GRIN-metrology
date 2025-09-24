import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
import torch
from typing import Optional, Any

def show_image(image: list | tuple | np.ndarray | torch.Tensor,
               figparams: Optional[dict[str, Any]] = None,
               imparams: Optional[dict[str, Any]] = None,
               savefigparams: Optional[dict[str, Any]] = None):
    

    figparams = figparams or {}
    imparams = imparams or {}
    
    array_image = to_array(image)
    plt.figure(**figparams)
    plt.imshow(array_image, **imparams)
    plt.tight_layout()
    plt.show()
    if savefigparams is not None:
        plt.savefig(**savefigparams)

def show_complex(image: list | tuple | np.ndarray | torch.Tensor,
                 savefigparams: Optional[dict[str, Any]] = None):
    
    array_image = to_array(image)
    fig, sub = plt.subplots(1,2)
    sub[0].imshow(np.abs(array_image))
    sub[0].set_title("Module")
    sub[1].imshow(np.angle(array_image))
    sub[1].set_title("Phase")
    plt.tight_layout()
    plt.show()
    if savefigparams is not None:
        plt.savefig(**savefigparams)

def to_array(image: list | tuple | np.ndarray | torch.Tensor):
    if isinstance(image, list | tuple):
        array_image = np.asarray(image)
    elif isinstance(image, torch.Tensor):
        array_image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array_image = np.copy(image)
    else:
        raise(TypeError("The image introduce does not match any of the compatible data types."))
    return array_image