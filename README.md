# atomic1D
## Contributions
This code is based on the OpenADAS analysis tool provided at [_cfe316/atomic_](https://github.com/cfe316/atomic).
## Purpose of this code
This code is intended to adapt the _atomic_ code (for atomic processes related to impurity seeding) to work with [_boutprojects/sd1d_](https://github.com/boutproject/SD1D).
This will be performed in three stages:
1. Read output of SD1D (i.e. `T_e(s)` and `n_e(s)`) into python. Output the radiatiated power as a function of position (i.e. `P_rad(s)`)
2. Incorporate `atomic1D` calculation of radiated power into `SD1D` (i.e. translate into C++) - calculate radiated power as a function of time and position, incorporate into the electron power balance
3. Incorporate `atomic1D` calculation of ionisation-stage distribution into `SD1D` - calculate ionisation distribution as a function of time and position, extend the electron power balance

In this approach `atomic1D` will be used to calculate the atomic physics processes while `SD1D` will be used to calculate impurity transport.

## Current status
* Have adapted the `fetch_adas_data` of `cfe316/atomic` to produce JSON database files from the .dat (fortran-77 formatted) files available from OpenADAS