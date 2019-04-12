# ISA Calculator
_Basic ISA Calculator for Python projects_

Contents:
  - Atmospheric Model up to 110km
  - Accurate Calculations for Temperature, Pressure and Density
  - Custom defined atmospheric conditions
  
  
With this module, it is possible to calculate, using the 1976 standard atmosphere model, the Temperature,
Density and Pressure at any point in the atmosphere from 0 up to 120,000 [m].

This package is useful for Aerospace and Aeronautical Engineers who wish to run simulations.

To use this package, follow these steps:

    - Install isacalc
    - Import isacalc as isa
    - Define the Atmosphere Model: isa.get_atmosphere()
    - It calculates, at the defined height:
        - Temperature
        - Pressure
        - Density
        - Speed of sound
        - Dynamic Viscosity

And thats it! An example Script:

    
    import isacalc as isa
    
    atmosphere = isa.get_atmosphere()
    
    h = 50000.0
    
    T, P, d, a, mu = isa.calculate_at_h(h, atmosphere)


  