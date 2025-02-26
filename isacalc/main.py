import json
from tabulate import tabulate
import numpy as np

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


    parser = ap.ArgumentParser(description='ISA Calculator', formatter_class=ap.ArgumentDefaultsHelpFormatter)

    settings_group = parser.add_argument_group("Program Settings", description="Change the program actions")
    settings_group.add_argument("-a", "--altitude", help="Altitude of single calculation", type=float)
    settings_group.add_argument("-t", "--table", help="Generate a table (start, stop, step)", nargs=3, type=float, action='store')

    table_group = parser.add_argument_group("Table Settings", description="Change the tabulating properties")
    table_group.add_argument("--params", help="Comma-separated list of parameters to tabulate. Options: T/P/D/A/M or 'all'", default='all')
    table_group.add_argument("-o", "--outfile", help="Output result to file", type=str)
    table_group.add_argument("--print", nargs='?', const='psql', help="Print table to console given a format as specified by the tabulate package", type=str, default=None)

    atmos_group = parser.add_argument_group("Atmosphere Settings", description="Change the atmospheric properties")
    atmos_group.add_argument("--infile", help="Path to JSON file containing atmospheric description", default=None)
    atmos_group.add_argument("--temps", help="List of temperatures (requires --heights)", default=None, type=comma_separated_floatlist)
    atmos_group.add_argument("--altitudes", help="List of altitudes (requires --temps)", default=None, type=comma_separated_floatlist)
    atmos_group.add_argument("--names", help="List of layer names (requires --heights and --temps)", default=None, type=comma_separated_stringlist)
    atmos_group.add_argument("-p", "--p0", help="Base Pressure in Pa", default=None, type=float)
    atmos_group.add_argument("-d", "--d0", help="Base Density in kg/m^3", default=None, type=float)
    atmos_group.add_argument('-g', '--g0', help="Gravitational acceleration in m/s^2", type=float, default=9.80665)
    atmos_group.add_argument('-R', '--R', help="Gas constant for air in J/(kg*K)", type=float, default=287.0)
    atmos_group.add_argument('-y', '--gamma', help="Specific heat capacity ratio for air", type=float, default=1.4)

    args = parser.parse_args()

    from isacalc import Atmosphere

    # loading from JSON file
    if args.infile is not None:
        with open(args.file, "r") as file:
            atmos_dict = json.load(file)
        atmosphere = Atmosphere(**atmos_dict)

    # loading from cmd args
    elif (args.p0 is not None
        or args.d0 is not None
        or args.temps is not None
        or args.altitudes is not None
        or args.g0 is not None
        or args.R is not None
        or args.gamma is not None):

        atmosphere = Atmosphere(
            p0=args.p0, d0=args.d0,
            Hn=args.altitudes, Tn=args.temps, Nn=args.names,
            g0=args.g0, R=args.R, gamma=args.gamma
        )

    # default
    else:
        atmosphere = Atmosphere()

    if args.table:
        start, stop, step = args.table
        if None in args.table:
            raise ap.ArgumentTypeError("start, stop and step need to all be defined to tabulate")

        table = atmosphere.tabulate(start, stop, step, args.outfile, args.params)

        if args.print is not None:
            print(tabulate(table, headers = 'keys', tablefmt = args.print))

    elif args.altitude:
        res = atmosphere.calculate(args.altitude)
        for val in res:
            print(val, end=", ")
        print()