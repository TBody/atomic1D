# Debugging trace on atomic-master/examples/radiation.py
# Note that this file is not expected to run!
# 
# A full trace of the examples/radiation.py code
# Will eventually need to translate all the core functionality of this into C++

# >> python -m pdb examples/radiation.py

atomic/examples/radiation.py(14)<module>()
-> import numpy as np
-> import atomic

-> ad = atomic.element('carbon')

--Call--
atomic/atomic/atomic_data.py(171)from_element(cls, element)
    """This is a variant constructor.
    It returns an instance of the class for a given element,
    looking up data values automatically. This is in contrast to the regular constructor,
    which requires you to already have the coeffiecients.

    Args:
        element: a string like 'Li' or 'lithium'
    Returns:
        An AtomicData class
    """
    # gets a list of filenames for 'ionisation', 'recombination', etc.
    -> @classmethod
    -> element_data = _element_data(element)

    --Call--
    atomic/atomic/atomic_data.py(74)_element_data(element)
    """Give a dictionary of ADF11 file names available for the given element.

    Args:
        element: a string like 'Li' or 'lithium'
    Returns:
        a dictionary of file names.

    {'ionisation' : 'scd96_li.dat',
    'recombination' : 'acd96_li.dat',
    ...
    }
    """
    e = element.lower()
    if e in ['c', 'carbon']:
        return carbon_data

        #carbon_data defined by
        atomic/atomic/atomic_data.py(68)
        -> carbon_data   = _element_data_dict(carbon_symbol, carbon_year, carbon_has_cx_power)

            # inputs to _element_data_dict defined by
            atomic/atomic/atomic_data.py(30)
            -> carbon_year = 96
            -> carbon_symbol = 'c'
            -> carbon_has_cx_power = True

            # _element_data_dict defined by
            --Call--
            atomic/atomic/atomic_data.py(55)_element_data_dict(el_symbol, el_year, has_cx_power=False)
                """Give a dictionary of ADF11 file names for the given element and year
                   and whether or not it has cx_power
                """
                -> data_dict = {}
                -> for key, value in datatype_abbrevs.items():
                ->     data_dict[key] = _make_filename(el_symbol, el_year, value)

                       --Call--
                       atomic/atomic/atomic_data.py(51)_make_filename(el_symbol, el_year, datatype)
                           """_make_filename('ne', 96, 'scd') -> 'scd96_ne.dat' """
                           -> return datatype + str(el_year) + '_' + el_symbol + '.dat'
                --End of call to _make_filename--
                data_dict['ionisation'] -> 'scd96_c.dat'

                -> if not has_cx_power: #False for carbon
                ->     data_dict.pop('cx_power', None)
                -> return data_dict

            --End of call to _element_data_dict--
            # carbon data returned as
            carbon_data -> {'ionisation'            : 'scd96_c.dat',
                            'recombination'         : 'acd96_c.dat',
                            'continuum_power'       : 'prb96_c.dat',
                            'line_power'            : 'plt96_c.dat',
                            'cx_power'              : 'prc96_c.dat',
                            'ionisation_potential'  : 'ecd96_c.dat'}
    --End of call to _element_data--
    element_data -> carbon_data

    # back to from_element
    atomic/atomic/atomic_data.py(186)from_element(cls, element)
    -> coefficients = {}
    -> for key, value in element_data.items():
    ->     fullfilename = _full_path(value)

           --Call--
           atomic/atomic/atomic_data.py(108)_full_path(file_)
               """ Figure out the location of an atomic datafile.
                   Files are all located in adas_data, which is at the same level
                   as the package directory, so to get there we must go up one.
               """
               # __file__ is the location of atomic_data.py
               -> module_path = os.path.dirname(os.path.realpath( __file__ ))
               -> return os.path.realpath(os.path.join(module_path, '..', 'adas_data', file_))
           --End of call to _full_path--
           fullfilename -> '/Users/thomasbody/Dropbox/Thesis/atomic/adas_data/scd96_c.dat'

    ->     coefficients[key] = RateCoefficient.from_adf11(fullfilename)
           --Call--
           atomic/atomic/atomic_data.py(237)from_adf11(cls, name)
               """Instantiate a RateCoefficient by reading in an adf11 file.
    
               Args:
                   name: The /full/name/of/an/adf11 file.
               """
               -> adf11_data = Adf11(name).read()
               --Call--
                   #Needs to first initialise the Adf11 object
                   --Call--
                   atomic/atomic/adf11.py(57)__init__(self,name)
                       -> if not os.path.isfile(name):
                       ->      raise IOError("no such file: '%s'" % name)
                       -> self.name = name
                   --End of call to Adf11.__init__--

                   #Then runs the method .read()
                   --Call--
                       atomic/atomic/adf11.py(63)read(self, class_=None)
                       # This is implemente in build_json.py - not expanded here
                       ->if class_ == None:
                       ->    self._sniff_class()
                       ->self._read_xxdata_11()
                       ->return self._convert_to_dictionary()
                   --End of call to Adf11.read--

               --End of call to read--
               adf11_data -> {'charge': 6, 'log_density': array([...]),
                'log_temperature': array([...]), 'number_of_charge_states': 6,
                'log_coeff': array([[[...]]]), 'class': 'scd', 'element': 'c',
                'name': '/Users/thomasbody/Dropbox/Thesis/atomic/adas_data/scd96_c.dat'}

               -> nuclear_charge = adf11_data['charge']
               -> element = adf11_data['element']
               -> filename = adf11_data['name']
               -> assert name == filename
               -> 
               -> log_temperature = adf11_data['log_temperature']
               -> log_density = adf11_data['log_density']
               -> log_coeff = adf11_data['log_coeff']
               -> 
               -> return cls(nuclear_charge, element, log_temperature, log_density,
               ->         log_coeff, name=filename)
                  --Call--
                  atomic/atomic/atomic_data.py/RateCoefficient(218)__init__(self, nuclear_charge, element,
                   log_temperature, log_density, log_coeff, name=None)
                      """Interpolation tables for the rate of some physical process.
                      
                      Contains one 2D spline interpolator for each charge state,
                      of an element, for one process like 'ionisation', 'recombination'.
                      
                      Attributes:
                          nuclear_charge (int): The element's Z.
                          element (str): Short element name like 'c' or 'ar'.
                          adf11_file (str): The /full/filename it came from.
                          log_temperature: np.array of log10 of temperature values
                          log_density: np.array of log10 of density values
                          log_coeff: a 3D np.array with shape (Z, temp, dens)
                          splines: list of scipy.interpolate.fitpack2.RectBivariateSpline
                              The list has length Z and is interpolations of log_coeff.
                      
                      NOTE: With the addition of ionisation_potentials, the RateCoefficient 
                      object is also storing tables of the ionisation potentials, even though
                      it is not at all the same thing as a rate coefficient. This is a kludge
                      for now.
                      """
                      -> self.nuclear_charge = nuclear_charge
                      -> self.element = element
                      -> self.adf11_file = name
                      -> 
                      -> self.log_temperature = log_temperature
                      -> self.log_density = log_density
                      -> self.log_coeff = log_coeff
                      -> 
                      -> self._compute_interpolating_splines()
                      --Call--
                      atomic/atomic/atomic_data.py/RateCoefficient(260)_compute_interpolating_splines(self)
                          # if we want to implement metastables there are more sets of coefficients
                          # than just for nuclear charge. This should be TODO'd.
                          # Also for stuff like ecd: ionisation potentials there is nuclear_charge + 1
                          -> self.splines = []
                          -> for k in range(self.nuclear_charge):
                          ->     x = self.log_temperature
                          ->     y = self.log_density
                          ->     z = self.log_coeff[k]
                          ->     self.splines.append(RectBivariateSpline(x, y, z))
                      --End of call to _compute_interpolating_splines--
                  --End of call to RateCoefficient.__init__--
           --End of call to from_adf11--
        coefficients['ionisation'] -> <atomic.atomic_data.RateCoefficient object at 0x109d8a470>
    -> return cls(coefficients)

    coefficients -> {'ionisation': <atomic.atomic_data.RateCoefficient object at 0x109d8a470>, ...}

--End of call to from_element--
ad -> <atomic.atomic_data.AtomicData object at 0x1117a1b70>

eq = atomic.CollRadEquilibrium(ad)
--Call--
atomic/atomic/collisional_radiative.py(8)__init__(self, atomic_data)
    -> self.atomic_data = atomic_data
    -> self.ionisation_coeff = atomic_data.coeffs['ionisation'] # RateCoefficient objects
    -> self.recombination_coeff = atomic_data.coeffs['recombination']
    -> self.nuclear_charge = atomic_data.nuclear_charge #could be generalized to include metastables?
--End of call to CollRadEquilibrium--
eq -> <atomic.collisional_radiative.CollRadEquilibrium object at 0x1117a1cf8>

temperature = np.logspace(0, 3, 50)
electron_density = 1e19

# y is a FractionalAbundance object.
y = eq.ionisation_stage_distribution(temperature, electron_density)
--Call--
atomic/atomic/collisional_radiative.py(14)ionisation_stage_distribution(self, temperature, density)
    """Compute ionisation stage fractions for collrad equilibrium.
    
    This case only includes ionisation and recombination.
    It does not include charge exchange, or any time-dependent effects.
    
    Args:
        temperature (array_like): temperatures [eV].
        density (array_like): densities [m^-3].
    
    Returns:
        A FractionalAbundance object
    """
    -> if len(temperature) == 1 and len(density) > 1:
    ->     temperature = temperature * np.ones_like(density)
    -> y = np.zeros((self.nuclear_charge + 1, len(temperature)))
    -> y[0] = np.ones_like(temperature)
    -> for k in range(self.nuclear_charge):
    ->     S = self.ionisation_coeff(k, temperature, density)
           --Call--
           atomic/atomic/atomic_data.py(271)__call__(self, k, Te, ne)
               """Evaulate the ionisation/recombination coefficients of
               k'th atomic state at a given temperature and density.
               
               Args:
                   k (int): Ionising or recombined ion stage,
                       between 0 and k=Z-1, where Z is atomic number.
                   Te (array_like): Temperature in [eV].
                   ne (array_like): Density in [m-3].
               
               Returns:
                   c (array_like): Rate coefficent in [m3/s].
               """
               -> c = self.log10(k, Te, ne)
               -> return np.power(10, c)
           --End of call to RateCoefficient.__call__--
    ->     alpha = self.recombination_coeff(k, temperature, density)
    ->     y[k+1] = y[k] * S / alpha
    -> 
    -> y /= y.sum(0) # fractional abundance
    -> return FractionalAbundance(self.atomic_data, y, temperature, density)
       --Call--
       # First calls init on a FractionalAbundance object
       atomic/atomic/abundance.py(17)__init__(self, atomic_data, y, temperature, density)
           """An array of ionisation stage fractions over density and/or temperature.

           Attributes:
               atomic_data (AtomicData): used by Radiation class
                   to get various coefficients.
               y (np.array, 2D): stores the fractional abundance of each
                   ionisation stage, k=0 to Z, for each point
                   of the given temperatures and densities.
                   Shape is (Z+1,x)
               temperature (array_like): list of temperatures [eV]
               density (array_like): list of densities [m^-3]
           """
           -> self.atomic_data = atomic_data
           -> self.y = y # fractional abundances of each charge state
           -> self.temperature = temperature
           -> self.density = density
       --End of call to FractionalAbundance--

--End of call to ionisation_stage_distribution--
y -> <atomic.abundance.FractionalAbundance object at 0x1171dcb38>

rad = atomic.Radiation(y, neutral_fraction=1e-2)
--Call--
atomic/atomic/radiation.py(20)__init__(self, ionisation_stage_distribution,
                                       impurity_fraction=1.,neutral_fraction=0.)
    """
    Attributes:
        y: a FractionalAbundance object.
        atomic_data: that FrAb's atomic data.
        temperature (np.array): that FrAb's temperature list.
        electron_density (np.array): that FrAb's density list.
        impurity_fraction: ???
            n_impurity = n_e * impurity_fraction
        neutral_fraction: fraction of neutral hydrogen, for cx_power.
            n_n = n_e * neutral_fraction
    """
    -> self.y = ionisation_stage_distribution
    -> self.atomic_data = ionisation_stage_distribution.atomic_data
    -> 
    -> self.temperature = self.y.temperature
    -> self.electron_density = self.y.density
    -> 
    -> self.impurity_fraction = impurity_fraction
    -> self.neutral_fraction = neutral_fraction
--End of call to Radiation--
rad -> <atomic.radiation.Radiation object at 0x1171dcc18>

import matplotlib.pyplot as plt
plt.figure(10); plt.clf()

lines = rad.plot()
--Call--
atomic/atomic/radiation.py(107)plot(self, **kwargs)
    """Plot the specific power for line_power, continuum_power,
    cx_power, and the total.

    Possible kwargs:
        'x': the x values of temperature for the plot.
            This will make the xscale linear.
        'ax': something about the axes; I don't understand yet.
    """
    -> if 'x' in kwargs:
    ->     xscale = 'linear'
    -> else:
    ->     xscale = 'log'
    -> 
    -> ax = kwargs.get('ax', plt.gca()) # gca is get current axes
    -> x = kwargs.get('x', self.temperature)
    -> 
    -> lines = []
    -> for key in ['line_power', 'continuum_power', 'cx_power', 'total']:
    ->     p = self.specific_power[key]
           --Retrieve property--
           atomic/atomic/radiation.py(35)specific_power(self)
               """Power per electron per impurity nucleus, [W m^3]"""
               -> power = self.power
               --Retrieve property--
                   -> return self._compute_power()
                   --Call--
                   atomic/atomic/radiation.py(66)_compute_power(self)
                        """
                        Compute radiation power density in [W/m3].
                        """
                        -> shape_ = self.atomic_data.nuclear_charge, self.temperature.shape[0]
                        -> 
                        -> power_coeffs = self._get_power_coeffs()
                           --Call--
                           atomic/atomic/radiation.py(49)_get_power_coeffs(self)
                               """Get a dict of RateCoefficient objects.
                               Looks for RateCoefficients called
                               line_power, continuum_power, and cx_power.
                               If one is not found, it returns a ZeroCoefficient.
                               
                               Returns:
                                   {'continuum_power': <RateCoefficient object>,
                                    'line_power'     : <RateCoefficient object>,
                                    'cx_power'       : <ZeroCoefficient object>}
                               """
                               -> power_coeffs = {}
                               -> for key in ['line_power', 'continuum_power', 'cx_power']:
                               ->     power_coeffs[key] = self.atomic_data.coeffs.get(key,
                               ->             ZeroCoefficient())
                                      --Call--
                                      atomic/atomic/atomic_data.py(328)__init__(self)
                                          """A subclass of RateCoefficient"""
                                          -> pass
                                      --End of call to ZeroCoefficient--

                               -> return power_coeffs
                           --End of call to _get_power_coeffs--
                           power_coeffs -> {'line_power': <atomic.atomic_data.RateCoefficient object at 0x1171dc588>,
                                            'continuum_power': <atomic.atomic_data.RateCoefficient object at 0x1171d41d0>,
                                            'cx_power': <atomic.atomic_data.RateCoefficient object at 0x1171dc710>}

                        -> radiation_power = {}
                        -> for key in power_coeffs.keys():
                        ->     radiation_power[key] = np.zeros(shape_)
                        -> 
                        -> ne = self.electron_density
                        -> ni = self.get_impurity_density()
                           --Call--
                           atomic/atomic/radiation.py(44)get_impurity_density(self)
                               return self.impurity_fraction * self.electron_density
                           --End of call to get_impurity_density--
                           ni -> 1e+19

                        -> n0 = self.get_neutral_density()
                           --Call--
                           atomic/atomic/radiation.py(47)get_neutral_density(self)
                               return self.neutral_fraction * self.electron_density
                           --End of call to get_neutral_density--
                           n0 -> 1e+17

                        -> y = self.y
                        -> 
                        -> for k in range(self.atomic_data.nuclear_charge):
                        ->     for key in radiation_power.keys():
                        ->         coeff = power_coeffs[key](k, self.temperature,
                        ->                 self.electron_density)
                                   --Call--
                                   atomic/atomic/atomic_data.py(271)__call__(self, k, Te, ne) #on RateCoefficient object
                                       """Evaulate the ionisation/recombination coefficients of
                                       k'th atomic state at a given temperature and density.

                                       Args:
                                           k (int): Ionising or recombined ion stage,
                                               between 0 and k=Z-1, where Z is atomic number.
                                           Te (array_like): Temperature in [eV].
                                           ne (array_like): Density in [m-3].

                                       Returns:
                                           c (array_like): Rate coefficent in [m3/s].
                                       """
                                       -> c = self.log10(k, Te, ne)
                                          --Call--
                                          atomic/atomic/atomic_data.py(287)log10(self, k, Te, ne)
                                              """Evaulate the logarithm of ionisation/recombination coefficients of
                                              k'th atomic state at a given temperature and density.

                                              If asked to evaluate for Te = np.array([1,2,3])
                                                  and ne = np.array([a,b,c]),
                                                  it will return coeffs at (1,a), (2,b), and (3,c),
                                                  not a 3x3 matrix of all the grid points.
                                                  I'm not sure yet if __call__ is typically
                                                  done with 1D or 2D arrays.

                                              Args:
                                                  k (int): Ionising or recombined ion stage.
                                                      Between 0 and k=Z-1, where Z is atomic number.
                                                  Te (array_like): Temperature in [eV].
                                                  ne (array_like): Density in [m-3].

                                              Returns:
                                                  c (array_like): log10(rate coefficent in [m3/s])
                                              """
                                              -> Te, ne = np.broadcast_arrays(Te, ne)
                                              -> log_temperature = np.log10(Te)
                                              -> log_density = np.log10(ne)

                                              -> c = self.splines[k](log_temperature, log_density, grid=False)
                                              -> return c
                                          --End of call to RateCoefficient.log10--
                                          c -> array([-34.28...-30.81188361])
                                       
                                       -> return np.power(10, c)

                                   --End of call to RateCoefficient.__call__--
                                   coeff -> array([  5.21...54211368e-31])
                        -> 
                        ->         if key in ['continuum_power', 'line_power']:
                        ->             if key in ['continuum_power']:
                        ->                 scale = ne * ni * y.y[k + 1]
                        ->             else:
                        ->                 scale = ne * ni * y.y[k]
                        ->         elif key in ['cx_power']:
                        ->             scale = n0 * ni * y.y[k]
                        -> 
                        ->         radiation_power[key][k] = scale * coeff
                        -> 
                        -> # compute the total power
                        -> radiation_power['total'] = reduce(lambda x,y: x+y,
                        ->         radiation_power.values())
                        -> 
                        -> # sum over all ionisation stages
                        -> for key in radiation_power.keys():
                        ->     radiation_power[key] = radiation_power[key].sum(0)
                        -> 
                        -> return radiation_power
                   --End of call to _compute_power--

               --End of retrieve power--
               power -> {'continuum_power': array([   56....845.7652292 ]),
                         'cx_power': array([  1.71...32935320e+04]),
                         'line_power': array([  3.49...92489704e+03]),
                         'total': array([  3.54...00641942e+04])}

               -> for key in power.keys():
               ->     power[key] /= self.electron_density * self.get_impurity_density()
                      --Call--
                      atomic/atomic/radiation.py(43)get_impurity_density(self)
                          -> return self.impurity_fraction * self.electron_density
                      --End of call to get_impurity_density--

               -> return power

           --End of retrieve specific_power--
           p -> {'continuum_power': array([  5.60...84576523e-35]),
                 'cx_power': array([  1.71...32935320e-34]),
                 'line_power': array([  3.49...92489704e-35]),
                 'total': array([  3.54...00641942e-34])}

    ->     l, = ax.semilogy(x, p, label=self._get_label(key))
           --Call--
           atomic/radiation.py(155)_get_label(self, key)
               """Called by plot"""
               -> labels = {
               ->     'continuum_power' : 'continuum',
               ->     'line_power' : 'line',
               ->     'cx_power' : 'charge-exchange',
               ->     'total' : 'total',
               -> }
               -> return labels.get(key, None)
           --End call to _get_label--
    ->     lines.append(l)
    -> 
    -> ax.set_xscale(xscale)
    -> ax.set_xlabel(r'$T_\mathrm{e}\ [\mathrm{eV}]$')
    -> ax.set_ylabel(r'$P\ [\mathrm{W/m^3}]$')
    -> 
    -> self._decorate_plot(ax, lines)
       --Call--
       atomic/radiation.py(155)_decorate_plot(self, ax, lines)
           """called by plot()
           Args:
               ax: the axes
               lines: different ax.semilogy (plots I suppose)?
           """
           -> alpha = 0.5 # transparency for fancy filling
           -> min_ = ax.get_ylim()[0]
           -> baseline = min_ * np.ones_like(self.temperature)
     
           -> for line in lines[:-1]:
           ->     x = line.get_xdata()
           ->     y = line.get_ydata()
           ->     ax.fill_between(x, y, baseline, color=line.get_color(), alpha=alpha)
           -> lines[-1].set_color('black')
       --End of call to _decorate_plot--

    -> plt.draw_if_interactive()
    -> 
    -> return lines

--End of call to Radiation.plot--

customize = True

if customize:
    plt.ylabel(r'$P/n_\mathrm{i} n_\mathrm{e}\ [\mathrm{W m^3}]$')
    plt.ylim(ymin=1e-35)
    
    # annotation
    s = '$n_0/n_\mathrm{e}$\n'
    if rad.neutral_fraction == 0:
        s += '$0$'
    else:
        # writes only 10^-1 or 10^-2, not 5.12 x 10^-1
        ne = rad.electron_density
        n0 = rad.get_neutral_density()
        --Call--
        atomic/atomic/radiation.py(47)get_neutral_density(self)
            return self.neutral_fraction * self.electron_density
        --End of call to get_neutral_density--

        exponent = np.log10(n0/ne)
        s += '$10^{%d}$' % exponent
    
    xy = (rad.temperature[-1], rad.specific_power['total'][-1])
    --Retrieve property--
    atomic/atomic/radiation.py(35)specific_power(self)
        """Power per electron per impurity nucleus, [W m^3]"""
        #See above
    --End of retrieve power--

    plt.annotate(s, xy, xytext=(1.05, 0.1),
        horizontalalignment='center',
        textcoords='axes fraction')

lines[-1].set_linewidth(2)
plt.legend(loc='best')

plt.draw()
plt.show()



