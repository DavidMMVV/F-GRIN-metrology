import matplotlib.pyplot as plt
import numpy as np

class Setup():
    def __init__(self, ) -> None:
        pass

if __name__ == "__main__":

    constant = 1000
    past_observable = 1
    x = np.linspace(-1, 1, 1000)
    
    sampling = (np.fft.fft(x), np.arange(constant))
    plt.plot(x)
    plt.show()