import numpy as np
from layers import NormalLayer, IsothermalLayer


class Atmosphere(object):

    def __init__(self):

        self.__p0 = 101325.0
        self.__d0 = 1.225

        self.__Nn = np.array(["Troposphere", "Tropopause", "Stratosphere", "Stratosphere", "Stratopause", "Mesosphere", "Mesosphere", "Mesopause", "Thermosphere", "Thermosphere", "Thermosphere"])
        self.__Tn = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95, 186.95, 201.95, 251.95])
        self.__Hn = np.array([0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 90000.0, 100000.0, 110000.0])
        self.__Lt = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1])

        self.__layers = []
        self.__build()

    def __build(self) -> None:

        p0, d0 = self.__p0, self.__d0

        for name, h0, h_i, T0, T_i, layer_type in zip(self.__Nn, self.__Hn[:-1], self.__Hn[1:], self.__Tn[:-1], self.__Tn[1:], self.__Lt):

            if layer_type == 1:

                Layer = NormalLayer(base_height=h0,
                                    base_temperature=T0,
                                    base_pressure=p0,
                                    base_density=d0,
                                    max_height=h_i,
                                    top_temperature=T_i,
                                    name=name)

                _, _, p0, d0 = Layer.get_ceiling_values()

            elif layer_type == 0:

                Layer = IsothermalLayer(base_height=h0,
                                        base_temperature=T0,
                                        base_pressure=p0,
                                        base_density=d0,
                                        max_height=h_i,
                                        name=name)

                _, _, p0, d0 = Layer.get_ceiling_values()

            else:
                raise ValueError

            self.__layers.append(Layer)

    def calculate(self, h) -> list:

        for idx in range(len(self.__layers)):

            if self.__Hn[idx] <= h < self.__Hn[idx + 1]:
                return self.__layers[idx].get_intermediate_values(h)

            elif h >= self.__Hn[idx + 1]:
                continue

