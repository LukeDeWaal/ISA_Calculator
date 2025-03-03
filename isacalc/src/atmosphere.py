from copy import copy
from typing import List, Set, Union, Iterable

import numpy as np
import sys, os, json

import pandas as pd

from .layers import NormalLayer, IsothermalLayer, LayerType, ptype

param_names = {
    'T': 'Temperature [K]',
    'P': 'Pressure [Pa]',
    'D': 'Density [kg/m^3]',
    'A' : 'Speed of Sound [m/s]',
    'M' : 'Dynamic Viscosity [kg/(m*s)]'
}

default_atmosphere = None

import importlib.resources
with importlib.resources.open_text("isacalc", "isa.json") as file:
    default_atmosphere = json.load(file)


class Atmosphere(object):

    def __init__(self, *args, **kwargs):
        """
        Atmosphere Object.

        :keyword g0     :   gravitational acceleration in m/s^2
        :keyword R      :   universal gas constant for air in J/(kg*K)
        :keyword gamma  :   specific-heat ratio for air
        :keyword p0     :   base pressure in Pa
        :keyword d0     :   base density in kg / m^3
        :keyword Tn     :   list of temperatures in K
        :keyword Hn     :   heights corresponding to given temperatures
        :keyword Nn     :   list of layer names

        :keyword infile :   path to JSON file containing atmospheric data
        """

        self.__g0: float = None
        self.__R: float = None
        self.__gamma: float = None

        self.__p0: float = None
        self.__d0: float = None
        self.__Nn: float = None
        self.__Tn: np.ndarray = None
        self.__Hn: np.ndarray = None

        self.__is_standard_atmosphere: bool = True

        self.__load(**kwargs)

        self.__Lt: np.ndarray = self.__get_lapse(self.__Hn, self.__Tn)

        self.__layers: np.ndarray = None
        self.__build()

        self.__last_calc_idx = 0
        self.__last_altimeter_idx = 0


    def __len__(self):
        """
        :return: Number of layers
        """
        return len(self.__layers)

    def __getitem__(self, idx):
        """
        Return Layer object at idx
        :param idx: index
        :return: layer object
        """
        return self.__layers[idx]

    def __str__(self):
        """
        Used for pretty-printing object info
        :return:
        """
        res = f"{'Standard' if self.__is_standard_atmosphere else 'Custom'} Atmosphere ({len(self)} layers):"
        res += f"\n * g0 = {self.__g0:<10.6f} [m/s^2]"
        res += f"\n * R  = {self.__R:<10.6f} [J/(kg*K)]"
        res += f"\n * y  = {self.__gamma:<10.6f} [-]"
        res += f"\n * p0 = {self.__p0:<10.3f} [Pa]"
        res += f"\n * d0 = {self.__d0:<10.6f} [kg/m^3]"
        for layer in self.__layers:
            res += '\n' + str(layer)
        return res

    def __repr__(self):
        """
        Used for pretty-printing object info
        :return:
        """
        return self.__str__()

    def calculate(self, h: float, *args, **kwargs) -> np.ndarray:
        """
        Calculate all atmospheric parameters at the given height
        :param h:               Altitude in meters
        :keyword start_index:   Start searching for the correct layer from given index.
                                Can be used to speed up calculations by skipping search.
                                Can result in ValueError if the relevant layer is skipped.
        :return:    Numpy Array containing Altitude, Temp, Press, Dens, SoundSpeed, DynVisc
        """

        start_idx = kwargs.get('start_index', 0)

        if h > self.__Hn[-1] or h < self.__Hn[0]:
            raise ValueError("Height is out of bounds")

        for self.__last_calc_idx in range(start_idx, len(self)):

            if abs(h - self.__Hn[self.__last_calc_idx + 1]) < 1e-3:
                return self.__layers[self.__last_calc_idx].ceiling_values

            elif abs(h - self.__Hn[self.__last_calc_idx]) < 1e-3:
                return self.__layers[self.__last_calc_idx].floor_values

            elif self.__Hn[self.__last_calc_idx] < h < self.__Hn[self.__last_calc_idx + 1]:
                return self.__layers[self.__last_calc_idx].get_values_at(h)

            elif h > self.__Hn[self.__last_calc_idx + 1]:
                continue

        raise ValueError("Failed to calculate")

    def altimeter(self, press: float, **kwargs) -> np.ndarray:
        """
        Calculate the altitude corresponding to given pressure and return all parameters
        :param press:   pressure in Pa
        :keyword start_index:   Start searching for the correct layer from given index.
                                Can be used to speed up calculations by skipping search.
                                Can result in ValueError if the relevant layer is skipped.
        :return: Numpy Array containing Altitude, Temp, Press, Dens, SoundSpeed, DynVisc
        """

        start_idx = kwargs.get('start_index', 0)

        if press > self.floor_values[ptype.PRESSURE.value] or press < self.ceiling_values[ptype.PRESSURE.value]:
            raise ValueError("Pressure is out of bounds")

        for self.__last_altimeter_idx in range(start_idx, len(self)):
            layer = self.__layers[self.__last_altimeter_idx]
            if layer.floor_values[ptype.PRESSURE.value] >= press > layer.ceiling_values[ptype.PRESSURE.value]:
                height = layer.get_height_from_pressure(press)
                return layer.get_values_at(height)

        raise ValueError("Failed to calculate")


    def densaltimeter(self, density: float, **kwargs) -> np.ndarray:
        """
        Calculate the altitude corresponding to given density and return all parameters
        :param density:   density in kg/m^3
        :keyword start_index:   Start searching for the correct layer from given index.
                                Can be used to speed up calculations by skipping search.
                                Can result in ValueError if the relevant layer is skipped.
        :return: Numpy Array containing Altitude, Temp, Press, Dens, SoundSpeed, DynVisc
        """

        start_idx = kwargs.get('start_index', 0)

        if density > self.floor_values[ptype.DENSITY.value] or density < self.ceiling_values[ptype.DENSITY.value]:
            raise ValueError("Density is out of bounds")

        for self.__last_altimeter_idx in range(start_idx, len(self)):
            layer = self.__layers[self.__last_altimeter_idx]
            if layer.floor_values[ptype.DENSITY.value] >= density > layer.ceiling_values[ptype.DENSITY.value]:
                height = layer.get_height_from_density(density)
                return layer.get_values_at(height)

        raise ValueError("Failed to calculate")


    def tabulate(self, start: float, stop: float, step: float, export_as: str = None, params: Iterable[str] = None, fastcalc: bool = True) -> pd.DataFrame:
        """
        Generate a table of atmospheric data
        :param start:       Table height start in meters
        :param stop:        Table height stop in meters (inclusive)
        :param step:        Table resolution in meters
        :param export_as:   Write table to csv/xlsx file
        :param params:      Set of parameters to tabulate with height (T/P/D/A/M or all)
        :param fastcalc:    Speed up the iterations by caching the last layer index
        :return:            Dataframe with the atmospheric table
        """

        if start is None or stop is None or step is None:
            raise RuntimeError("start, stop and step cannot be None")

        if params is None or (isinstance(params, str) and params.lower() == 'all'):
            params = ['T', 'P', 'D', 'A', 'M']
        else:
            if isinstance(params, list) or (isinstance(params, str) and params.lower() != 'all'):
                params = [p.upper() for p in params]

        heights = np.arange(start=start, stop=stop+step, step=step)
        table_shape = (len(heights), len(params)+1)
        result = np.zeros(table_shape, dtype=float)

        if fastcalc:
            self.__last_calc_idx = 0

        for idx, height in enumerate(heights):

            if fastcalc:
                parameters = self.calculate(height, start_index=self.__last_calc_idx)
            else:
                parameters = self.calculate(height)

            result[idx,:] = parameters

        df_result = pd.DataFrame(data=result,
                                 index=range(0, table_shape[0]),
                                 columns=['Height [m]'] + [param_names[p] for p in params])

        if export_as is not None:
            extension = export_as.split('.')[-1].lower()

            if extension == 'csv':
                df_result.to_csv(export_as, index_label='Index')

            else:
                df_result.to_excel(export_as, index_label='Index')

        return df_result

    def export_json(self, path: str) -> None:
        """
        Export current atmosphere as JSON object
        :param path:
        :return:
        """
        res = {
            "g0": self.__g0,
            "R": self.__R,
            "gamma": self.__gamma,
            "p0": self.__p0,
            "d0": self.__d0,
            "Hn": list(self.__Hn),
            "Tn": list(self.__Tn),
        }
        with open(path, 'w+') as fp:
            json.dump(res, fp, indent=4)

    @property
    def temperatures(self) -> np.ndarray:
        """
        Array of temperatures along layers
        :return:
        """
        return self.__Tn

    @property
    def altitudes(self) -> np.ndarray:
        """
        Array of altitudes of layers
        :return:
        """
        return self.__Hn

    @property
    def floor_values(self) -> np.ndarray:
        """
        Dictionary with the atmospheric base values
        :return:
        """
        return self.__layers[0].floor_values

    @property
    def ceiling_values(self) -> np.ndarray:
        """
        Dictionary with the atmospheric base values
        :return:
        """
        return self.__layers[-1].ceiling_values

    @property
    def constants(self) -> dict:
        """
        Dictionary of physical constants of atmosphere
        :return:
        """
        return {
            'g0': self.__g0,
            'R': self.__R,
            'gamma': self.__gamma
        }

    @property
    def height_boundaries(self):
        """
        Method to calculate for which range the atmosphere model can be used
        :return: Min, Max Height
        """
        return self.__Hn[0], self.__Hn[-1]

    @staticmethod
    def __get_lapse(Hn, Tn) -> np.ndarray:
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

        return np.array(types)

    def __load(self, **kwargs):
        """
        Helper method to load the atmosphere model
        """
        classname = type(self).__name__
        infile = kwargs.get('infile', None)
        if infile is not None:
            self.__is_standard_atmosphere = False
            with open(infile, "r") as file:
                atmosphere = json.load(file)
            for key in atmosphere.keys():
                setattr(self, f"_{classname}__{key}", atmosphere[key])
        else:
            for key in default_atmosphere.keys():
                if key in kwargs and kwargs[key] is not None:
                    self.__is_standard_atmosphere = False
                    setattr(self, f"_{classname}__{key}", kwargs[key])
                elif key in kwargs and kwargs[key] is None:
                    setattr(self, f"_{classname}__{key}", default_atmosphere[key])
                else:
                    if key in kwargs and key == 'Nn' and kwargs[key] is None:
                        setattr(self, f"_{classname}__{key}", "Noname")
                    else:
                        setattr(self, f"_{classname}__{key}", default_atmosphere[key])

    def __build(self) -> None:
        """
        Helper method to build the atmosphere object
        """

        self.__Tn = np.array(self.__Tn)
        self.__Hn = np.array(self.__Hn)
        self.__layers = []

        p0, d0 = self.__p0, self.__d0

        for name, h0, h_i, T0, T_i, layer_type in zip(self.__Nn, self.__Hn[:-1], self.__Hn[1:], self.__Tn[:-1], self.__Tn[1:], self.__Lt):

            if layer_type == LayerType.STANDARD:

                layer = NormalLayer(base_height=h0,
                                    base_temperature=T0,
                                    base_pressure=p0,
                                    base_density=d0,
                                    max_height=h_i,
                                    top_temperature=T_i,
                                    name=name,
                                    g0=self.__g0,
                                    R=self.__R,
                                    gamma=self.__gamma)

                h, T0, p0, d0, a0, mu0 = layer.ceiling_values

            elif layer_type == LayerType.ISOTHERMAL:

                layer = IsothermalLayer(base_height=h0,
                                        base_temperature=T0,
                                        base_pressure=p0,
                                        base_density=d0,
                                        max_height=h_i,
                                        name=name,
                                        g0=self.__g0,
                                        R=self.__R,
                                        gamma=self.__gamma)

                h, T0, p0, d0, a0, mu0 = layer.ceiling_values

            else:
                raise ValueError

            self.__layers.append(layer)

        self.__layers = np.array(self.__layers)
