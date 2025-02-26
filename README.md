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
atmosphere = isa.Atmosphere()

print(atmosphere)

h = 50000.0

# Calculate a single time 
T, P, d, a, mu = atmosphere.calculate(h)

# Generate a table
table = atmosphere.tabulate(0, 25000, 100)
```
