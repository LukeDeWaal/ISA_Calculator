import sys
from copy import copy
from typing import List, Tuple, Dict, Union, Set, Optional
from enum import Enum

import numpy as np


class LayerType(Enum):
    STANDARD = 1
    ISOTHERMAL = 2

class ptype(Enum):
    HEIGHT = 0
    TEMPERATURE = 1
    PRESSURE = 2
    DENSITY = 3
    SPEEDOFSOUND = 4
    DYN_VISCOSITY = 5


class Layer(object):
    """
    Template Object for Layers
    """

    def __init__(self, base_height: float, base_temperature: float, base_pressure: float, base_density: float,
                 max_height: float, name: str = '', g0: float = 9.80665, R: float = 287.0, gamma: float = 1.4, *args,
                 **kwargs):
        """
        All Properties of a layer, excluding its type, which will be expressed
        by making separate objects for the different layers
        :param base_height:         Height at which the layer starts
        :param base_temperature:    Temperature at the bottom of the layer
        :param base_pressure:       Pressure at the bottom of the layer
        :param base_density:        Density at the bottom ofthe layer
        :param max_height:          Height up to which the layer extends
        """

        self._g0: float = g0
        self._R: float = R
        self._gamma: float = gamma

        self.__h0: float = base_height
        self.__T0: float = base_temperature
        self.__p0: float = base_pressure
        self.__d0: float = base_density
        self.__h_max: float = max_height

        if name:
            self.__name: str = name
        else:
            self.__name: str = "unnamed"

        self._cache: Dict[str, Optional[np.ndarray]] = {
            'ceil': None,
            'floor': None,
        }

    def __str__(self):
        """
        Used for pretty-printing object info
        :return:
        """
        return f"{self.__name:<16s} | {self.__h0:10.3f} - {self.__h_max:10.3f} [m] | {self.floor_values[0]:6.2f} - {self.ceiling_values[0]:6.2f} [K]"

    @staticmethod
    def sutherland_viscosity(T: float, mu0: float = 1.716e-5, T0: float = 273.15, S: float = 110.4) -> float:
        """
        Method to calculate the dynamic viscosity of air
        :param T:   Temperature at which to calculate
        :param mu0: Reference viscosity
        :param T0:  Reference temperature
        :param S:   Sutherland Temperature
        :return:    Viscosity of air in kg/(m*s)
        """
        return mu0 * np.power((T / T0), 1.5) * (T0 + S) / (T + S)

    def speed_of_sound(self, temp: float) -> float:
        """
        Method to calculate the speed of sound at a certain temperature
        :param temp: Temperature in K
        :return: speed of sound in m/s
        """
        return np.sqrt(self._gamma * self._R * temp)

    @property
    def floor_values(self) -> np.ndarray:
        """
        Getter function to obtain the hidden layer states
        :return: List of all base values
        """
        if self._cache['floor'] is None:
            self._cache['floor'] = np.array([
                self.__h0,
                self.__T0,
                self.__p0,
                self.__d0,
                self.speed_of_sound(self.__T0),
                self.sutherland_viscosity(self.__T0)
            ])
        return self._cache['floor']

    @property
    def ceiling_height(self) -> float:
        """
        Getter function to obtain the maximum height of the layer
        :return: Maximum height
        """
        return self.__h_max

    @property
    def floor_height(self) -> float:
        """
        Getter function to obtain the height at which the layer starts
        :return: Base Height
        """
        return self.__h0

    @property
    def name(self) -> str:
        """
        Method to return the name of the layer
        :return: name
        """
        return copy(self.__name)

    @property
    def ceiling_values(self) -> None:
        """
        This method will be overridden
        """
        raise NotImplementedError("This method should be overridden")

    def get_values_at(self, h: float) -> None:
        """
        This method will be overridden
        """
        raise NotImplementedError("This method should be overridden")

    def get_height_from_pressure(self, pressure: float) -> None:
        """
        This method will be overridden
        """
        raise NotImplementedError("This method should be overridden")

    def get_height_from_density(self, density: float) -> None:
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

    @property
    def ceiling_values(self) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at the ceiling of the layer
        :return: temperature, pressure, density, speed of sound
        """

        if self._cache['ceil'] is None:
            h0, T0, P0, D0, a0, mu0 = self.floor_values
            h = self.ceiling_height

            P = P0 * np.exp(-self._g0 / (self._R * T0) * (h - h0))
            D = D0 * np.exp(-self._g0 / (self._R * T0) * (h - h0))

            a = self.speed_of_sound(T0)
            mu = self.sutherland_viscosity(T0)

            self._cache['ceil'] = np.array([h, T0, P, D, a, mu])

        return self._cache['ceil']

    def get_values_at(self, h: float) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at height h, between the base and ceiling of the layer
        :param h: Height at which to evaluate the temperature, pressure, density
        :return: temperature, pressure, density, speed of sound
        """
        h_max = self.ceiling_height
        h0 = self.floor_height

        if h > h_max:
            raise ValueError(f"Given height exceeds layer height: {round(h, 3)} > {round(h_max, 3)}")

        elif h < h0:
            raise ValueError(f"Given height is too low for given layer:  {round(h, 3)} < {round(h0, 3)}")

        else:
            if abs(h - h0) < 1e-3:
                return self.floor_values

            elif abs(h - h_max) < 1e-3:
                return self.ceiling_values

            else:
                h0, T0, P0, D0, a0, mu0 = self.floor_values

                P = P0 * np.exp(-self._g0 / (self._R * T0) * (h - h0))
                D = D0 * np.exp(-self._g0 / (self._R * T0) * (h - h0))

                a = self.speed_of_sound(T0)
                mu = self.sutherland_viscosity(T0)

                return np.array([h, T0, P, D, a, mu])

    def get_height_from_pressure(self, pressure: float) -> float:
        """
        Calculate the altitude from a given pressure if inside this layer
        :param pressure: pressure in Pa
        :return: altitude in m
        """

        floor_temp = self.floor_values[ptype.TEMPERATURE.value]
        floor_press = self.floor_values[ptype.PRESSURE.value]
        ceil_press  = self.ceiling_values[ptype.PRESSURE.value]

        if floor_press >= pressure > ceil_press:

            return self.floor_height - np.log(pressure / floor_press)*(self._R * floor_temp) / self._g0

        else:
            raise ValueError(f"Given pressure is outside layer: {pressure} ! [{floor_press}, {ceil_press}]")

    def get_height_from_density(self, density: float) -> None:
        """
        Calculate the altitude from a given density if inside this layer
        :param density: density in kg/m^3
        :return: altitude in m
        """

        floor_temp = self.floor_values[ptype.TEMPERATURE.value]
        floor_dens = self.floor_values[ptype.DENSITY.value]
        ceil_dens  = self.ceiling_values[ptype.DENSITY.value]

        if floor_dens >= density > ceil_dens:

            return self.floor_height - np.log(density / floor_dens)*(self._R * floor_temp) / self._g0

        else:
            raise ValueError(f"Given density is outside layer: {density} ! [{floor_dens}, {ceil_dens}]")

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

    @property
    def ceiling_values(self) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at the ceiling of the layer
        :return: temperature, pressure, density, speed of sound
        """

        h0, T0, P0, D0, a0, mu0 = self.floor_values
        h = self.ceiling_height

        L = (self.__T_top - T0) / (h - h0)
        C = -self._g0 / (L * self._R)  # To Simplify and shorten code we define the following expression for the exponent

        P = P0 * (self.__T_top / T0) ** C
        D = D0 * (self.__T_top / T0) ** (C - 1)

        a = self.speed_of_sound(self.__T_top)
        mu = self.sutherland_viscosity(self.__T_top)

        return np.array([h, self.__T_top, P, D, a, mu])

    def get_values_at(self, h: float) -> np.ndarray:
        """
        Method to get the temperature, pressure and density at height h, between the base and ceiling of the layer
        :param h: Height at which to evaluate the temperature, pressure, density
        :return: temperature, pressure, density, speed of sound
        """

        h_max = self.ceiling_height
        h0 = self.floor_height

        if h > h_max:
            raise ValueError(f"Given height exceeds layer height: {round(h, 3)} > {round(h_max, 3)}")

        elif h < h0:
            raise ValueError(f"Given height is too low for given layer:  {round(h, 3)} < {round(h0, 3)}")

        else:
            if abs(h - h0) < 1e-3:
                return self.floor_values

            elif abs(h - h_max) < 1e-3:
                return self.ceiling_values

            else:
                h0, T0, P0, D0, a0, mu0 = self.floor_values

                L = (self.__T_top - T0) / (h_max - h0)
                C = -self._g0 / (L * self._R)  # To Simplify and shorten code we define the following expression for the exponent

                T = T0 + L * (h - h0)

                P = P0 * np.power((T / T0), C)
                D = D0 * np.power((T / T0), (C - 1))

                a = self.speed_of_sound(T)
                mu = self.sutherland_viscosity(T)

                return np.array([h, T, P, D, a, mu])

    def get_height_from_pressure(self, pressure: float) -> float:
        """
        Calculate the altitude from a given pressure if inside this layer
        :param pressure: pressure in Pa
        :return: altitude in m
        """

        floor_temp = self.floor_values[ptype.TEMPERATURE.value]
        floor_press = self.floor_values[ptype.PRESSURE.value]
        ceil_press  = self.ceiling_values[ptype.PRESSURE.value]

        L = (self.__T_top - floor_temp) / (self.ceiling_height - self.floor_height)
        C = -self._g0 / (L * self._R)

        if floor_press >= pressure > ceil_press:
            return float(self.floor_height + (floor_temp / L) * (np.power((pressure / floor_press), 1/C) - 1))

        else:
            raise ValueError(f"Given pressure is outside layer: {pressure} ! [{floor_press}, {ceil_press}]")

    def get_height_from_density(self, density: float) -> float:
        """
        Calculate the altitude from a given density if inside this layer
        :param density: density in kg/m^3
        :return: altitude in m
        """
        floor_temp = self.floor_values[ptype.TEMPERATURE.value]
        floor_dens = self.floor_values[ptype.DENSITY.value]
        ceil_dens  = self.ceiling_values[ptype.DENSITY.value]

        L = (self.__T_top - floor_temp) / (self.ceiling_height - self.floor_height)
        C = -self._g0 / (L * self._R)

        if floor_dens >= density > ceil_dens:
            return float(self.floor_height + (floor_temp / L) * (np.power((density / floor_dens), 1/(C-1)) - 1))

        else:
            raise ValueError(f"Given density is outside layer: {density} ! [{floor_dens}, {ceil_dens}]")