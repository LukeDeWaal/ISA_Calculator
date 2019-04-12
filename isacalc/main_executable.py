from isacalc.src import Atmosphere

def get_atmosphere() -> Atmosphere:
    """
    Function to obtain the atmosphere model
    :return: Atmospheric Model
    """
    return Atmosphere()


def calculate_at_h(h: float, atmosphere_model: Atmosphere = get_atmosphere()) -> list:
    """
    Function to calculate Temperature, Pressure and Density at h
    :param h:                   Height in [m]
    :param atmosphere_model:    Atmosphere Object
    :return:                    [h, T, P, D]
    """
    return atmosphere_model.calculate(h)


