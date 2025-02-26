# isacalc : A Basic ISA Calculator for Python projects

## Contents:
  - Accurate Atmospheric Model up to 110km
  - Accurate Calculations for Temperature, Pressure and Density
  - Option for user-defined atmospheric conditions
  - Tabulate your data to save future computation time
  - Save tabulated data in .csv or .xlsx files
  
  
With this module, it is possible to calculate, using the 1976 standard atmosphere model, the Temperature,
Density and Pressure at any point in the atmosphere from 0 up to 110,000 \[m].

This package is useful for Aerospace and Aeronautical Engineers who wish to run simulations using atmospheric data.

To use this package, follow these steps:

- Install isacalc
- Import isacalc as isa
- Define the Atmosphere Model: isa.get_atmosphere()
- Calculate all parameters at a cetain height: isa.calculate_at_h(h, atmosphere_model)
- It calculates, at the defined height:
    - Temperature
    - Pressure
    - Density
    - Speed of sound
    - Dynamic Viscosity

And thats it! An example Script:

    
```python
import isacalc as isa
    
atmosphere = isa.Atmosphere()

h = 50000.0

# Calculate a single time 
T, P, d, a, mu = atmosphere.calculate(h)

# Generate a table
table = atmosphere.tabulate(0, 25000, 100)

```


  
