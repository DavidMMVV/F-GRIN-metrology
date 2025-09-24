import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.signal import convolve2d


ARRAY_CLASS = Union[list, np.ndarray]

class Layer:
    """
    Simple layer with a certain distribution of refraction index
    """
    def __init__(self, z:float,  n:float | complex | np.ndarray = 1, shape: tuple = (0,0), extent:tuple = (0,2,0,1)):

        self.z = z
        self.extent = extent

        if type(n) == np.ndarray:
            shape = np.shape(n)
            self.n_arr = np.complex128(n)
        else:
            self.n_arr = np.complex128(n * np.ones(shape))

        self.shape = shape
        self.Y, self.X = np.mgrid[extent[0]:extent[1]:shape[0]*1j,
                                  extent[2]:extent[3]:shape[1]*1j]

    def __add__(self, layer2: "Layer"):
        layer1 = Layer(self.z, self.n_arr, extent=self.extent)
        return Multilayer((layer1, layer2))
    
    def add_circle(self, n: complex, radious:float, center:tuple):
        condition = ((self.Y - center[0])**2 + (self.X - center[1])**2) <= radious**2
        self.n_arr[condition] = n

    def add_rectangle(self, n: complex, center: tuple, dimensions: tuple):
        condition_1 = (self.Y <= (center[0] + dimensions[0])) & (self.Y >= (center[0] - dimensions[0]))
        condition_2 = (self.X <= (center[1] + dimensions[1])) & (self.X >= (center[1] - dimensions[1]))
        self.n_arr[condition_1&condition_2] = n

    def convolve(self, kernel=np.ones((9,9))/9**2):
        self.n_arr = convolve2d(self.n_arr, kernel, mode='same')

    def show(self):

        fig, sub = plt.subplots(1,2, sharex=True, sharey=True)
        fig.suptitle("Index of refraction of the {:.03e}mm deep layer".format(self.z))
        p1 = sub[0].imshow(np.real(self.n_arr), extent=self.extent, cmap="jet")
        fig.colorbar(p1, ax=sub[0], shrink=0.5)
        sub[0].set_title("Real part")
        sub[0].set_xlabel("x(mm)")
        sub[0].set_ylabel("y(mm)")
        p2 = sub[1].imshow(np.imag(self.n_arr), extent=self.extent, cmap="jet")
        fig.colorbar(p2, ax=sub[1], shrink=0.5)
        sub[1].set_title("Imaginary part")
        sub[1].set_xlabel("x(mm)")
        plt.tight_layout()
        #plt.show()
    

class Multilayer:
    """
    This object contains multiple layers with different index of refraction distributions.
    """
    def __init__(self, layers: tuple):
        n_arr = []
        z_arr = []
        for i in range(len(layers)):
            n_arr.append(layers[i].n_arr)
            z_arr.append(layers[i].z)
        self.n_arr = np.array(n_arr)
        self.z_arr = np.array(z_arr)

class Field():
    """
    This class includes all the characteristics of a field and its possible pransformations.
    """
    def __init__(self, field: Union[complex, np.ndarray] = 1, wavelength:float = 600e-9, shape: tuple = (0,0), extent: tuple = (0,2,0,1)):
        self.wavelength = wavelength
        self.extent = extent

        if type(field) == np.ndarray:
            shape = np.shape(field)
            self.field_arr = field
        else:
            self.field_arr = field * np.ones(shape)

        self.shape = shape
        self.Y, self.X = np.mgrid[extent[0]:extent[1]:shape[0]*1j,
                                  extent[2]:extent[3]:shape[1]*1j]

    def propagate(self, layer: Layer, propagator):
        """
        This function propagates an input field across a layer.

        Args:
            input_field (np.ndarray): Field which enters to the layer.
            layer (Layer):
            propaggator: Function which propagates the field. 
        """
        output_field = propagator(self.field_arr, layer.n_arr, layer.z, self.wavelength)
        self.field_arr = output_field

    def show(self, title: str = "Field"):
            
            fig, sub = plt.subplots(1,2, sharex=True, sharey=True)
            fig.suptitle(title)
            p1 = sub[0].imshow(np.abs(self.field_arr), extent=self.extent, cmap="jet")
            fig.colorbar(p1, ax=sub[0], shrink=0.5)
            sub[0].set_title("Module")
            sub[0].set_xlabel("x(mm)")
            sub[0].set_ylabel("y(mm)")
            p2 = sub[1].imshow(np.angle(self.field_arr), extent=self.extent, cmap="jet")
            fig.colorbar(p2, ax=sub[1], shrink=0.5)
            sub[1].set_title("Phase")
            sub[1].set_xlabel("x(mm)")
            plt.tight_layout()
            #plt.show()

def propagate(input_field: Field, layer: Layer, propagator):
    """
    This function propagates an input field across a layer.

    Args:
        input_field (Field): Field which enters to the layer.
        layer (Layer): Layer through which the field will be propagated.
        propaggator: Function which propagates the field. 
    
    Returns:
        output_field (Field): Field which leaves the layer.
    """
    output_field = propagator(input_field.field_arr, layer.n_arr, layer.z, input_field.wavelength)

    return Field(output_field, input_field.wavelength, input_field.shape, input_field.extent)

def first_approach_propagator(U_i, n, z: float, wavelength: float, *coordinates: ARRAY_CLASS):
    """_summary_

    Args:
        U_i (np.ndarray): Input field.
        n (_type_): Index of refraction.
        z (float): Distance of propagation.
        wavelength (float): Wavelength of the wave.
        *coordinates (ARRAY_CLASS): 2D arrays with the coordinates along each axes in a tuple as (Y, X).

    Returns:
        U_o (np.ndarray): Output field.
    """

    return U_i * np.exp(1j * np.pi * (n**2 - 1) * z / wavelength)

def loss_MSE(y_guess, y_measured):
    """
    Mean square error loss function.

    Args:
        y_guess: Value guessed.
        y_measured: Real vaue.
    Returns:
        score: The final float number which estimates how far we are from the real solution.
    """
    return np.mean((np.real(y_guess)-np.real(y_measured))**2 + (np.imag(y_guess)-np.imag(y_measured))**2)

def back_first_approach(U_measured, input_field: Field, layer:Layer, propagator, epsilon = 1):
    """
    This function retun the corrected values of index of refraction 

    Args:
        U_measured (_type_): _description_
        input_field (Field): _description_
        layer (Layer): _description_
        propagator (_type_): _description_
        epsilon (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    U_estim = propagate(input_field, layer, propagator).field_arr
    dL_dU = (2 * (U_estim - U_measured))
    dU_dn = U_estim * (2j * layer.n_arr) * np.pi * layer.z / input_field.wavelength
    grad = dL_dU * dU_dn
    layer.n_arr -= epsilon * np.real(grad)
    return layer

def optimize(U_measured, input_field: Field, initial_layer:Layer, propagator, n_iter: int, min_loss: float = 1e-3, epsilon=1):
    layer = Layer(initial_layer.z, initial_layer.n_arr, initial_layer.shape, initial_layer.extent)
    loss_arr = []
    for i in range(n_iter):
        U_estim = propagate(input_field, layer, propagator).field_arr
        loss = np.abs(np.real(loss_MSE(U_estim, U_measured)))
        loss_arr.append(loss)
        layer = back_first_approach(U_measured, input_field, layer, propagator, epsilon)
        if i % (n_iter // 10) == 0:
            print("{}/{}: Loss: {:.04f}".format(i, n_iter, loss))
        if loss <= min_loss:
            break
    plt.figure()
    plt.plot(np.arange(i+1), loss_arr)
    plt.xlabel("Iteration")
    plt.ylabel("Loss function")
    #plt.show()
    return layer

LENGTH_mm = 100
HEIGHT_mm = 100

def main():
    extent = (-LENGTH_mm/2,LENGTH_mm/2,-HEIGHT_mm/2,HEIGHT_mm/2)
    wavelength = 600e-9
    shape = (500,500)
    z1 = 10e-2
    z2 = 20e-3


    p1 = Layer(z1, 1, shape, extent)
    p1.add_rectangle(1.5, (0,0), (4,10))
    p1.add_circle(1.2, 20, (0,-10))
    p1.add_circle(1.2, 20, (0,10))
    p1.convolve(np.ones((10,10))/10**2)
    #p1.show()

    p2 = Layer(z2, 1, shape, extent)
    p2.add_rectangle(1.5, (0,0), (10,4))
    p2.add_circle(1.2, 5, (-10,0))
    p2.add_circle(1.2, 5, (10,0))
    p2.convolve(np.ones((10,10))/10**2)
    #p2.show()

    total = p1 + p2

    U_i = Field(1, wavelength, shape, extent)
    U_1 = propagate(U_i, p1, first_approach_propagator)
    U_1.show("Field after first layer")

    init_layer = Layer(z1, 1, shape, extent)
    n_iter = 1000
    min_loss = 1e-8
    epsilon =  1e-8 * wavelength / (2 * np.pi * init_layer.z)

    guess = optimize(U_1.field_arr, U_i, init_layer, first_approach_propagator, n_iter, min_loss, epsilon)
    guess.n_arr = 2 - guess.n_arr
    guess.show()
    p1.show()
    plt.show()


    

if __name__ == "__main__":
    main()