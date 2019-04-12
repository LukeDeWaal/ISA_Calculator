# ISA Calculator
Basic ISA Calculator for Python projects

Contents:
  - Atmospheric Model up to 120km
  - Accurate Calculations for Temperature, Pressure and Density
  
  
With this module, it is possible to calculate, using the 1976 standard atmosphere model, the Temperature,
Density and Pressure at any point in the atmosphere from 0 up to 120,000 [m].

This package is useful for Aerospace and Aeronautical Engineers who wish to run simulations.

To use this package, follow these steps:

    - Install isacalc
    - Import isacalc as isa
    - Define the Atmosphere Model: isa.get_atmosphere()
    - Calculate the Temperature, Pressure and Density at height h using isa.calculate_at_h(h, atmosphere_model)

And thats it! An example Script:

    
    import isacalc as isa
    
    atmosphere = isa.get_atmosphere()
    h = 50000.0
    
    h, T, P, d = isa.calculate_at_h(h, atmosphere)


Planned Future Features:

    - Atmosphere Model Customization
    - Mach number and viscosity calculations
  