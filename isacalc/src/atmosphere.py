from typing import List, Set, Union, Iterable

import numpy as np
import sys, os, json

import pandas as pd

from .layers import NormalLayer, IsothermalLayer, LayerType

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

        self.__g0 = None
        self.__R = None
        self.__gamma = None

        self.__p0 = None
        self.__d0 = None
        self.__Nn = None
        self.__Tn = None
        self.__Hn = None

        classname = type(self).__name__
        for key in default_atmosphere.keys():
            if key in kwargs and kwargs[key] is not None:
                setattr(self, f"_{classname}__{key}", kwargs[key])
            elif key in kwargs and kwargs[key] is None:
                setattr(self, f"_{classname}__{key}", default_atmosphere[key])
            else:
                if kwargs and key == 'Nn':
                    setattr(self, f"_{classname}__{key}", "Noname")
                else:
                    setattr(self, f"_{classname}__{key}", default_atmosphere[key])

        self.__Lt = self.__get_lapse(self.__Hn, self.__Tn)

        self.__layers = []
        self.__build()
        self.__last_idx = 0

    def calculate(self, h: float, *args, **kwargs) -> np.ndarray:
        """
        Calculate all atmospheric parameters at the given height
        :param h:               Altitude in meters
        :keyword start_index:   Start searching for the correct layer from given index.
                                Can be used to speed up calculations by skipping search.
                                Can result in ValueError if the relevant layer is skipped.
        :return:    Numpy Array containing Temp, Press, Dens, SoundSpeed, DynVisc
        """

        start_idx = kwargs.get('start_index', 0)

        if h > self.__Hn[-1] or h < self.__Hn[0]:
            raise ValueError("Height is out of bounds")

        for self.__last_idx in range(start_idx, len(self.__layers)):

            if abs(h - self.__Hn[self.__last_idx +1]) < 1e-3:
                return self.__layers[self.__last_idx ].get_ceiling_values()

            elif abs(h - self.__Hn[self.__last_idx ]) < 1e-3:
                return self.__layers[self.__last_idx ].get_base_values()

            elif self.__Hn[self.__last_idx ] < h < self.__Hn[self.__last_idx  + 1]:
                return self.__layers[self.__last_idx ].get_intermediate_values(h)

            elif h > self.__Hn[self.__last_idx  + 1]:
                continue

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
            self.__last_idx = 0

        for idx, height in enumerate(heights):

            if fastcalc:
                parameters = self.calculate(height, start_index=self.__last_idx)
            else:
                parameters = self.calculate(height)

            result[idx,0] = height
            result[idx,1:] = parameters

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

                T0, p0, d0, a0, mu0 = layer.get_ceiling_values()

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

                T0, p0, d0, a0, mu0 = layer.get_ceiling_values()

            else:
                raise ValueError

            self.__layers.append(layer)

        self.__layers = np.array(self.__layers)

    @staticmethod
    def __load_default() -> dict:
        try:
            from importlib import resources as impresources
        except ImportError:
            # Try backported to PY<37 `importlib_resources`.
            import importlib_resources as impresources

        try:
            inp_file = (impresources.files('isacalc.data') / 'isa.json')
            with inp_file.open("r") as file:  # or "rt" as text file with universal newlines
                default_atmosphere = json.load(file)
        except AttributeError:
            # Python < PY3.9, fall back to method deprecated in PY3.11.
            default_atmosphere = json.load(impresources.open_text('isacalc.data', 'isa.json'))
        return default_atmosphere