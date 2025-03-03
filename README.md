# ISACALC : A Basic ISA Calculator for Python projects

  - Accurate Atmospheric Model up to 110km
  - Accurate Calculations for Temperature, Pressure and Density
  - Option for user-defined atmospheric conditions:
    - Change the temperature gradients
    - Change the base pressure or density
    - Change physical constants such as gravity or universal gas constant for air
    - Make your custom atmosphere and save it to a JSON file to be loaded
  - Tabulate your data to save future computation time
    - Start, Stop and Step in meters
    - Save tabulated data in .csv or .xlsx files
  
With this module, it is possible to calculate, using the 1976 standard atmosphere model, the Temperature,
Density and Pressure at any point in the atmosphere from 0 up to 110,000 \[m].

This package is useful for Aerospace and Aeronautical Engineers who wish to run simulations using atmospheric data.

To install this package, simply `pip install isacalc`

And thats it! An simple example script:

```python
import isacalc as isa
    
# Default atmosphere
std_atm = isa.Atmosphere()

print(std_atm)
"""
    Standard Atmosphere (10 layers):
    * g0 = 9.806650   [m/s^2]
    * R  = 287.000000 [J/(kg*K)]
    * y  = 1.400000   [-]
    * p0 = 101325.000 [Pa]
    * d0 = 1.225000   [kg/m^3]
    Troposphere      |      0.000 -  11000.000 [m] |   0.00 - 11000.00 [K]
    Tropopause       |  11000.000 -  20000.000 [m] | 11000.00 - 20000.00 [K]
    Stratosphere     |  20000.000 -  32000.000 [m] | 20000.00 - 32000.00 [K]
    Stratosphere     |  32000.000 -  47000.000 [m] | 32000.00 - 47000.00 [K]
    Stratopause      |  47000.000 -  51000.000 [m] | 47000.00 - 51000.00 [K]
    Mesosphere       |  51000.000 -  71000.000 [m] | 51000.00 - 71000.00 [K]
    Mesosphere       |  71000.000 -  84852.000 [m] | 71000.00 - 84852.00 [K]
    Mesopause        |  84852.000 -  90000.000 [m] | 84852.00 - 90000.00 [K]
    Thermosphere     |  90000.000 - 100000.000 [m] | 90000.00 - 100000.00 [K]
    Thermosphere     | 100000.000 - 110000.000 [m] | 100000.00 - 110000.00 [K]
"""

# Tweak parameters in the constructor
custom_atm = isa.Atmosphere(p0=100000.0, g0=9.81, gamma=1.42)
print(custom_atm)
"""
    Custom Atmosphere (10 layers):
    * g0 = 9.810000   [m/s^2]
    * R  = 287.000000 [J/(kg*K)]
    * y  = 1.420000   [-]
    * p0 = 100000.000 [Pa]
    * d0 = 1.225000   [kg/m^3]
    Troposphere      |      0.000 -  11000.000 [m] |   0.00 - 11000.00 [K]
    Tropopause       |  11000.000 -  20000.000 [m] | 11000.00 - 20000.00 [K]
    Stratosphere     |  20000.000 -  32000.000 [m] | 20000.00 - 32000.00 [K]
    Stratosphere     |  32000.000 -  47000.000 [m] | 32000.00 - 47000.00 [K]
    Stratopause      |  47000.000 -  51000.000 [m] | 47000.00 - 51000.00 [K]
    Mesosphere       |  51000.000 -  71000.000 [m] | 51000.00 - 71000.00 [K]
    Mesosphere       |  71000.000 -  84852.000 [m] | 71000.00 - 84852.00 [K]
    Mesopause        |  84852.000 -  90000.000 [m] | 84852.00 - 90000.00 [K]
    Thermosphere     |  90000.000 - 100000.000 [m] | 90000.00 - 100000.00 [K]
    Thermosphere     | 100000.000 - 110000.000 [m] | 100000.00 - 110000.00 [K]
"""

# Export your custom atmosphere model to JSON for easy import later
custom_atm.export_json('my_atm.json')

# Load an atmosphere model from a JSON file
custom_atm = isa.Atmosphere(infile='my_atm.json')

# Generate a pandas dataframe table and optionally export is as a csv or xlsx file
table = custom_atm.tabulate(
    start=0, stop=25000, step=100, export_as='my_atm.csv'
)

# Do simple calculations
# NOTE: if the given values are outside the atmosphere model, will raise ValueError
params = custom_atm.calculate(h=660.0)          # Get params at height
params = custom_atm.altimeter(press=26000.0)    # Get params at pressure
params = custom_atm.densaltimeter(density=0.85) # Get params at density
ceiling_params  = custom_atm.ceiling_values     # Get the params at max altitude
floor_params    = custom_atm.floor_values       # Get the params at lowest altitude

# Do calculations within a specific layer
# NOTE: if the given values are outside the layer, will raise ValueError
layer = custom_atm[0]                           # Take the first layer (Troposphere)
params = layer.get_values_at(300)               # Get params at height within layer
height = layer.get_height_from_pressure(100000) # Get height at pressure within layer
height = layer.get_height_from_density(1.2)     # Get height at density within layer

ceiling_params  = layer.ceiling_values          # Get params at layer's max altitude
floor_params    = layer.floor_values            # Get params at layer's min altitude

```

To define a custom atmosphere in JSON format, create a file with the structure shown below.
Any missing parameters will be taken from the standard atmosphere.
The JSON file defining the standard atmosphere is as follows:
```json
{
  "g0": 9.80665,
  "R": 287.0,
  "gamma": 1.4,
  "p0": 101325.0,
  "d0": 1.225,
  "Hn": [
    0,
    11000.0,
    20000.0,
    32000.0,
    47000.0,
    51000.0,
    71000.0,
    84852.0,
    90000.0,
    100000.0,
    110000.0
  ],
  "Tn": [
    288.15,
    216.65,
    216.65,
    228.65,
    270.65,
    270.65,
    214.65,
    186.95,
    186.95,
    201.95,
    251.95
  ],
  "Nn": [
    "Troposphere",
    "Tropopause",
    "Stratosphere",
    "Stratosphere",
    "Stratopause",
    "Mesosphere",
    "Mesosphere",
    "Mesopause",
    "Thermosphere",
    "Thermosphere",
    "Thermosphere"
  ]
}
```