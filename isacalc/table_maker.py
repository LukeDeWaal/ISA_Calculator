import numpy as np
import pandas as pd
from isacalc.src import Atmosphere


def __extract_desired_params(boolean_list: list, parameter_list: list):
    """
    Function which takes 2 lists, one with booleans and one with any other data types.
    Returns a list with all values of the parameter list which had the same indices as
    the True booleans in the boolean list
    :param boolean_list:    List of True or False
    :param parameter_list:  List of parameters
    :return: List of parameters that coincide with True items in boolean list
    """

    if len(boolean_list) != len(parameter_list):
        raise IndexError("Input sizes do not match")

    result = []

    for boolean, item in zip(boolean_list, parameter_list):
        if boolean is True:
            result.append(item)

    return result


def tabulate(height_range: tuple, atmosphere_model: Atmosphere = Atmosphere(), export_as: str = None, params: list or str = None) -> pd.DataFrame:
    """
    Function to tabulate all the calculated data.
    For large projects where the ISA data needs to be calculated and accessed a lot, this will be
    a beneficial tradeoff of space against time. Storing in a csv or xlsx file is optional
    :param height_range:        Range of heights, (start, stop, step) format is required
    :param atmosphere_model:    Model on which the computations need to be performed
    :param export_as:           If left blank, will not be saved to file
    :param params:              List of all desired parameters to be recorded
    :return:                    Numpy Array of all values calculated [Height, Temp, Press, Dens, SOS, Visc]
    """

    if params is None:
        params = []

    if isinstance(params, str) and params.lower() == 'all':
        params = ['t', 'p', 'd', 'a', 'mu']

    param_names = ['Temperature [K]',
                   'Pressure [Pa]',
                   'Density [kg/m^3]',
                   'Speed of Sound [m/s]',
                   'Dynamic Viscosity [kg/(m*s)]']

    start, stop, step = height_range
    heights = np.arange(start=start, stop=stop+step, step=step)

    params = [a.lower() for a in params if type(a) == str and len(params) != 0]

    boolean_record_list = [True]*5

    if params:
        if 't' not in params:
            boolean_record_list[0] = False

        if 'p' not in params:
            boolean_record_list[1] = False

        if 'd' not in params:
            boolean_record_list[2] = False

        if 'a' not in params:
            boolean_record_list[3] = False

        if 'mu' not in params:
            boolean_record_list[4] = False

    table_shape = (len(heights), sum(boolean_record_list)+1)
    result = np.zeros(table_shape, dtype=float)

    for idx, height in enumerate(heights):

        parameters = __extract_desired_params(boolean_record_list, list(atmosphere_model.calculate(height)))
        result[idx,:] = [height] + parameters

    param_names = __extract_desired_params(boolean_record_list, param_names)

    df_result = pd.DataFrame(data=result,
                             index=range(0, table_shape[0]),
                             columns=['Height [m]']+param_names)

    if export_as:

        extension = export_as.split('.')[-1].lower()

        if extension == 'csv':
            df_result.to_csv(export_as, index=False)

        else:
            df_result.to_excel(export_as)

    return df_result


if __name__ == "__main__":

    table = tabulate((0, 10000, 100), export_as=r'test1.csv', params='all')
