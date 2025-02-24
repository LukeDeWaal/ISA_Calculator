import json

import numpy as np

from isacalc.src import Atmosphere, calculate_at_h
from isacalc.table_maker import tabulate




if __name__ == "__main__":

    import argparse as ap
    listlen = None

    def comma_separated_floatlist(arg: str):
        values = arg.split(",")

        global listlen

        if listlen is None:
            listlen = len(values)
        else:
            if listlen != len(values):
                raise ap.ArgumentTypeError("Received lists of different lengths")

        return np.array([float(v) for v in values])

    def comma_separated_stringlist(arg: str or None):

        global listlen

        if arg is None and listlen is None:
            return np.array(["Unnamed" for _ in range(100)])
        elif arg is None and listlen is not None:
            return np.array(["Unnamed" for _ in range(listlen)])

        values = arg.split(",")
        return np.array([str(v) for v in values])


    parser = ap.ArgumentParser(description='ISA Calculator')

    settings_group = parser.add_argument_group("Program Settings", description="Change the program actions")
    settings_group.add_argument("-a", "--altitude", help="Altitude of single calculation", type=float)
    settings_group.add_argument("-t", "--table", help="Generate a table", action='store_true')
    settings_group.add_argument("-o", "--outfile", help="Output result to file", type=str)

    table_group = parser.add_argument_group("Table Settings", description="Change the tabulating properties")
    table_group.add_argument("--start", help="Table start altitude in meters", type=float)
    table_group.add_argument("--stop", help="Table stop altitude in meters (inclusive)", type=float)
    table_group.add_argument("--step", help="Table altitude step in meters", type=float)
    table_group.add_argument("--params", help="Comma-separated list of parameters to tabulate. Options: t/p/d/a/mu or 'all'", default='all')


    atmos_group = parser.add_argument_group("Atmosphere Settings", description="Change the atmospheric properties")
    atmos_group.add_argument("-p", "--p0", help="Base Pressure in Pa", default=None, type=float)
    atmos_group.add_argument("-d", "--d0", help="Base Density in kg/m^3", default=None, type=float)
    atmos_group.add_argument("--temps", help="List of temperatures (requires --heights)", default=None, type=comma_separated_floatlist)
    atmos_group.add_argument("--altitudes", help="List of altitudes (requires --temps)", default=None, type=comma_separated_floatlist)
    atmos_group.add_argument("--names", help="List of layer names (requires --heights and --temps)", default=None, type=comma_separated_stringlist)
    atmos_group.add_argument("--file", help="Path to JSON file containing atmospheric description",default=None)

    args = parser.parse_args()

    # loading from JSON file
    if args.file is not None:
        with open(args.file, "r") as file:
            atmos_dict = json.load(file)
        atmosphere = Atmosphere(**atmos_dict)

    # loading from cmd args
    elif (args.p0 is not None
        and args.d0 is not None
        and args.temps is not None
        and args.altitudes is not None):

        atmosphere = Atmosphere(p0=args.p0, d0=args.d0, Hn=args.altitudes, Tn=args.temps, Nn=args.names)

    # default
    else:
        atmosphere = Atmosphere()

    if args.table:
        if args.start is None or args.stop is None or args.step is None:
            raise ap.ArgumentTypeError("start, stop and step need to all be defined to tabulate")

        table = tabulate((args.start, args.stop, args.step), atmosphere, args.outfile, args.params)

    else:
        res = calculate_at_h(args.altitude, atmosphere)
        print(res)