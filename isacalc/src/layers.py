import sys
from copy import copy
from typing import List, Tuple, Dict, Union, Set, Optional
from enum import Enum

import numpy as np


class LayerType(Enum):
    STANDARD    = 1
    ISOTHERMAL  = 2


class Layer(object):
    """
    Template Object for Layers
    """

    def __init__(self, base_height: float, base_temperature: float, base_pressure: float, base_density: float,
                 max_height: float, name: str = '', g0: float = 9.80665, R: float = 287.0, gamma: float = 1.4, *args, **kwargs):
        """
        All Properties of a layer, excluding its type, which will be expressed
        by making separate objects for the different layers
        :param base_height:         Height at which the layer starts
        :param base_temperature:    Temperature at the bottom of the layer
        :param base_pressure:       Pressure at the bottom of the layer
        :param base_density:        Density at the bottom ofthe layer
        :param max_height:          Height up to which the layer extends
        """

        self.g0 = g0
        self.R = R
        self.gamma = gamma

        self.__h0 = base_height
        self.__T0 = base_temperature
        self.__p0 = base_pressure
        self.__d0 = base_density
        self.__h_max = max_height

        self.__name = name

    @staticmethod
    def sutherland_viscosity(T, mu0=1.716e-5, T0=273.15, S=110.4) -> float:
        """
        Method to calculate the dynamic viscosity of air
        :param T:   Temperature at which to calculate
        :param mu0: Reference viscosity
        :param T0:  Reference temperature
        :param S:   Sutherland Temperature
        :return:    Viscosity of air in kg/(m*s)
        """
        return mu0 * np.power((T/T0), 1.5) * (T0 + S)/(T + S)


    def speed_of_sound(self, temp: float) -> float:
        """
        Method to calculate the speed of sound at a certain temperature
        :param temp: Temperature in K
        :return: speed of sound in m/s
        """
        return np.sqrt(self.gamma*self.R*temp)

    def get_base_values(self) -> np.ndarray:
        """
        Getter function to obtain the hidden layer states
        :return: List of all base values
        """
        return np.array([self.__T0, self.__p0, self.__d0, self.speed_of_sound(self.__T0), self.sutherland_viscosity(self.__T0)])

    def get_ceiling_height(self) -> float:
        """
        Getter function to obtain the maximum height of the layer
        :return: Maximum height
        """
        return self.__h_max

    def get_base_height(self) -> float:
        """
        Getter function to obtain the height at which the layer starts
        :return: Base Height
        """
        return self.__h0

    def get_name(self) -> str:
        """
        Method to return the name of the layer
        :return: name
        """
        return copy(self.__name)

    def get_ceiling_values(self) -> None:
        """
        This method will be overridden
        """
        raise NotImplementedError("This method should be overridden")

    def get_intermediate_values(self, h) -> None:
        """
        This method will be overridden
        """
        raise NotImplementedError("This method should be overridden")


class IsothermalLayer(Layer):

    def __init__(self, base_height: float,
                 base_temperature: float,
                 base_pressure: float,
                 base_density: float,
                 max_height: float,
                 name: str = '',
                 **kwargs):

        super().__init__(base_height=base_height,
                         base_temperature=base_temperature,
                         base_pressure=base_pressure,
                         base_density=base_density,
                         max_height=max_height,
                         name=name,
                         **kwargs)

    def get_ceiling_values(self) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at the ceiling of the layer
        :return: temperature, pressure, density, speed of sound
        """

        T0, P0, D0, a0, mu0 = self.get_base_values()
        h = self.get_ceiling_height()
        h0 = self.get_base_height()

        P = P0 * np.exp(-self.g0 / (self.R * T0) * (h - h0))
        D = D0 * np.exp(-self.g0 / (self.R * T0) * (h - h0))

        a = self.speed_of_sound(T0)
        mu = self.sutherland_viscosity(T0)

        return np.array([T0, P, D, a, mu])

    def get_intermediate_values(self, h) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at height h, between the base and ceiling of the layer
        :param h: Height at which to evaluate the temperature, pressure, density
        :return: temperature, pressure, density, speed of sound
        """
        h_max = self.get_ceiling_height()
        h0 = self.get_base_height()

        ret = np.zeros((5,), dtype=float)

        if h > h_max:
            raise ValueError(f"Given height exceeds layer height: {round(h, 3)} > {round(h_max, 3)}")

        if h < h0:
            raise ValueError(f"Given height is too low for given layer:  {round(h, 3)} < {round(h0, 3)}")

        if h == h0:
            return np.array(self.get_base_values())

        T0, P0, D0, a0, mu0 = self.get_base_values()

        P = P0 * np.exp(-self.g0 / (self.R * T0) * (h - h0))
        D = D0 * np.exp(-self.g0 / (self.R * T0) * (h - h0))

        a = self.speed_of_sound(T0)
        mu = self.sutherland_viscosity(T0)

        return np.array([T0, P, D, a, mu])


class NormalLayer(Layer):

    def __init__(self, base_height: float,
                 base_temperature: float,
                 base_pressure: float,
                 base_density: float,
                 max_height: float,
                 top_temperature: float,
                 name: str = '',
                 **kwargs):

        super().__init__(base_height=base_height,
                         base_temperature=base_temperature,
                         base_pressure=base_pressure,
                         base_density=base_density,
                         max_height=max_height,
                         name=name,
                         **kwargs)

        self.__T_top = top_temperature

    def get_ceiling_values(self) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at the ceiling of the layer
        :return: temperature, pressure, density, speed of sound
        """

        T0, P0, D0, a0, mu0 = self.get_base_values()
        h = self.get_ceiling_height()
        h0 = self.get_base_height()

        L = (self.__T_top - T0) / (h - h0)
        C = -self.g0 / (L * self.R)  # To Simplify and shorten code we define the following expression for the exponent

        P = P0 * (self.__T_top / T0) ** C
        D = D0 * (self.__T_top / T0) ** (C - 1)

        a = self.speed_of_sound(self.__T_top)
        mu = self.sutherland_viscosity(self.__T_top)

        return np.array([self.__T_top, P, D, a, mu])

    def get_intermediate_values(self, h) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at height h, between the base and ceiling of the layer
        :param h: Height at which to evaluate the temperature, pressure, density
        :return: temperature, pressure, density, speed of sound
        """

        h_max = self.get_ceiling_height()
        h0 = self.get_base_height()

        if h > h_max:
            raise ValueError

        if h == h0:
            return self.get_base_values()

        T0, P0, D0, a0, mu0 = self.get_base_values()

        L = (self.__T_top - T0) / (h_max - h0)
        C = -self.g0 / (L * self.R)  # To Simplify and shorten code we define the following expression for the exponent

        T = T0 + L*(h - h0)

        P = P0 * np.power((T / T0) , C)
        D = D0 * np.power((T / T0) , (C - 1))

        a = self.speed_of_sound(T)
        mu = self.sutherland_viscosity(T)

        return np.array([T, P, D, a, mu])

