import numpy as np
import sys, os
from .layers import NormalLayer, IsothermalLayer
import json

from enum import Enum

class LayerType(Enum):
    STANDARD = 1
    ISOTHERMAL = 2

class Atmosphere(object):

    def __init__(self, *args, **kwargs):

        with open("isa.json", "r") as file:
            default_atmosphere = json.load(file)

        self.__p0 = None
        self.__d0 = None
        self.__Nn = None
        self.__Tn = None
        self.__Hn = None

        classname = type(self).__name__
        for key in default_atmosphere.keys():
            if key in kwargs and kwargs[key] is not None:
                setattr(self, f"_{classname}__{key}", kwargs[key])
            else:
                if kwargs and key != 'Nn':
                    raise ValueError(f"Failed to retrieve value for '{key}'")
                elif kwargs and key == 'Nn':
                    setattr(self, f"_{classname}__{key}", "Noname")
                else:
                    setattr(self, f"_{classname}__{key}", default_atmosphere[key])

        self.__Lt = self.__get_lapse(self.__Hn, self.__Tn)

        self.__layers = []
        self.__build()

    def get_height_boundaries(self):
        """
        Method to calculate for which range the atmosphere model can be used
        :return: Min, Max Height
        """
        return self.__Hn[0], self.__Hn[-1]

    @staticmethod
    def __get_lapse(Hn, Tn) -> list:
        """
        Static Method to calculate the layer types of all layers
        :param Hn: Heights
        :param Tn: Temperatures
        :return: Layer Types
        """

        types = []

        for i in range(len(Hn)-1):
            delta_T = Tn[i+1] - Tn[i]

            lapse = delta_T/(Hn[i+1] - Hn[i])

            if lapse != 0:
                if abs(delta_T) > 0.5:
                    types.append(LayerType.STANDARD)

                else:
                    types.append(LayerType.ISOTHERMAL)

            elif lapse == 0:
                types.append(LayerType.ISOTHERMAL)

        return types

    def __build(self) -> None:
        """
        Helper method to build the atmosphere object
        """

        p0, d0 = self.__p0, self.__d0

        for name, h0, h_i, T0, T_i, layer_type in zip(self.__Nn, self.__Hn[:-1], self.__Hn[1:], self.__Tn[:-1], self.__Tn[1:], self.__Lt):

            if layer_type == LayerType.STANDARD:

                Layer = NormalLayer(base_height=h0,
                                    base_temperature=T0,
                                    base_pressure=p0,
                                    base_density=d0,
                                    max_height=h_i,
                                    top_temperature=T_i,
                                    name=name)

                T0, p0, d0, a0, mu0 = Layer.get_ceiling_values()

            elif layer_type == LayerType.ISOTHERMAL:

                Layer = IsothermalLayer(base_height=h0,
                                        base_temperature=T0,
                                        base_pressure=p0,
                                        base_density=d0,
                                        max_height=h_i,
                                        name=name)

                T0, p0, d0, a0, mu0 = Layer.get_ceiling_values()

            else:
                raise ValueError

            self.__layers.append(Layer)

    def calculate(self, h: float) -> np.ndarray:
        """
        Calculate all atmospheric parameters at the given height
        :param h:
        :return:
        """

        if h > self.__Hn[-1] or h < self.__Hn[0]:
            raise ValueError("Height is out of bounds")

        for idx in range(len(self.__layers)):

            if h == self.__Hn[idx+1]:
                return self.__layers[idx].get_ceiling_values()

            elif h == self.__Hn[idx]:
                return self.__layers[idx].get_base_values()

            elif self.__Hn[idx] < h < self.__Hn[idx + 1]:
                return self.__layers[idx].get_intermediate_values(h)

            elif h > self.__Hn[idx + 1]:
                continue

        raise ValueError("Failed to calculate")

def calculate_at_h(h: float, atmosphere_model: Atmosphere = Atmosphere()) -> np.ndarray:
    """
    Function to calculate Temperature, Pressure and Density at h
    :param h:                   Height in [m]
    :param atmosphere_model:    Atmosphere Object
    :return:                    [h, T, P, D]
    """
    return atmosphere_model.calculate(h)