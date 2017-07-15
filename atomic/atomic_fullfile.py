# __init__.py
from __future__ import absolute_import
from .atomic_data import AtomicData
from .collisional_radiative import CollRadEquilibrium
from .time_dependent_rates import RateEquations, RateEquationsWithDiffusion
from .radiation import Radiation
from .electron_cooling import ElectronCooling

element = AtomicData.from_element

# atomic_data
from __future__ import absolute_import
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .adf11 import Adf11

datatype_abbrevs = {
        'ionisation' : 'scd',
        'recombination' : 'acd',
        'continuum_power' : 'prb',
        'line_power' : 'plt',
        'cx_power' : 'prc',
        'ionisation_potential' : 'ecd',
}

# The system of registering element names, symbols, years,
# and what datatypes they have could certainly be improved. However it works at the moment.
# maybe with a pandas DataFrame?

lithium_year = 96
lithium_symbol = 'li'
#it might be nice to get a prc data! doesn't seem to be available

argon_year = 89
argon_symbol = 'ar'
argon_has_cx_power = True

carbon_year = 96
carbon_symbol = 'c'
carbon_has_cx_power = True

nitrogen_year = 96
nitrogen_symbol = 'n'

neon_year = 96
neon_symbol = 'ne'

beryllium_year = 96
beryllium_symbol = 'be'

boron_year = 89
boron_symbol = 'b'
boron_has_cx_power = True

# imaginary element, for testing purposes
imaginarium_year = 0
imaginarium_data = {}

def _make_filename(el_symbol, el_year, datatype):
    """_make_filename('ne', 96, 'scd') -> 'scd96_ne.dat' """
    return datatype + str(el_year) + '_' + el_symbol + '.dat'

def _element_data_dict(el_symbol, el_year, has_cx_power=False):
    """Give a dictionary of ADF11 file names for the given element and year
       and whether or not it has cx_power
    """
    data_dict = {}
    for key, value in datatype_abbrevs.items():
        data_dict[key] = _make_filename(el_symbol, el_year, value)
    if not has_cx_power:
        data_dict.pop('cx_power', None)
    return data_dict

lithium_data  = _element_data_dict(lithium_symbol,  lithium_year)
argon_data    = _element_data_dict(argon_symbol,    argon_year,  argon_has_cx_power)
carbon_data   = _element_data_dict(carbon_symbol,   carbon_year, carbon_has_cx_power)
nitrogen_data = _element_data_dict(nitrogen_symbol, nitrogen_year)
neon_data     = _element_data_dict(neon_symbol,     neon_year)
beryllium_data= _element_data_dict(beryllium_symbol,beryllium_year)
boron_data    = _element_data_dict(boron_symbol,    boron_year)

def _element_data(element):
    """Give a dictionary of ADF11 file names available for the given element.

    Args:
        element: a string like 'Li' or 'lithium'
    Returns:
        a dictionary of file names.

    {'ionisation' : 'scd96_li.dat',
    'recombination' : 'acd96_li.dat',
    ...
    }

    This could presumably be made more general, especially with automated lookup of files.
    """
    e = element.lower()
    if e in ['li', 'lithium']:
        return lithium_data
    elif e in ['c', 'carbon']:
        return carbon_data
    elif e in ['ar', 'argon']:
        return argon_data
    elif e in ['ne', 'neon']:
        return neon_data
    elif e in ['n', 'nitrogen']:
        return nitrogen_data
    elif e in ['be', 'beryllium']:
        return beryllium_data
    elif e in ['b', 'boron']:
        return boron_data
    else:
        raise NotImplementedError('unknown element: %s' % element)


def _full_path(file_):
    """ Figure out the location of an atomic datafile.
        Files are all located in adas_data, which is at the same level
        as the package directory, so to get there we must go up one.
    """
    # __file__ is the location of atomic_data.py
    module_path = os.path.dirname(os.path.realpath( __file__ ))
    return os.path.realpath(os.path.join(module_path, '..', 'adas_data', file_))


class AtomicData(object):
    """
    Attributes:
        element (string): the element symbol, like 'Li' or 'C'
        coeffs (dict): a dictionary of RateCoefficient objects
            with keys like 'ionisation'
        nuclear_charge (int) : the element's Z

    """

    def __init__(self, coefficients):
        """
        Args:
            coefficients (dict):
                The rate coefficients, as RateCoefficient objects.
        """
        self.coeffs = coefficients
        self._check_consistency()
        self._make_element_initial_uppercase()

    def copy(self):
        """Make a new object that is a copy of this one.

        This is used in pec.py's TransitionPool.
        """
        new_coeffs = {}
        #iteritems() is a method on dicts that gives an iterator
        for key, value in self.coeffs.items():
            new_coeffs[key] = value.copy()

        return self.__class__(new_coeffs)

    def _check_consistency(self):
        """Add the nuclear_charge and element attributes.

        Each individual RateCoefficient object has its own copy of the
        nuclear_charge, say 3 for Li. Because a set() can only contain one copy
        of the number 3, testing for len == 1 ensures that all the files
        correspond to the same nucleus.
        """
        nuclear_charge = set()
        element = set()
        #coeff are the RateCoefficient objects
        for coeff in self.coeffs.values():
            nuclear_charge.add(coeff.nuclear_charge)
            element.add(coeff.element.lower())

        assert len(nuclear_charge) == 1, 'inconsistent nuclear charge.'
        assert len(element) == 1, 'inconsistent element name.'

        self.nuclear_charge = nuclear_charge.pop()
        self.element = element.pop()

    @classmethod
    def from_element(cls, element):
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
        element_data = _element_data(element)

        coefficients = {}
        for key, value in element_data.items():
            fullfilename = _full_path(value)
            coefficients[key] = RateCoefficient.from_adf11(fullfilename)

        return cls(coefficients)

    def _make_element_initial_uppercase(self):
        e = self.element
        self.element = e[0].upper() + e[1:]

class RateCoefficient(object):
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
    def __init__(self, nuclear_charge, element, log_temperature, log_density,
            log_coeff, name=None):
        self.nuclear_charge = nuclear_charge
        self.element = element
        self.adf11_file = name

        self.log_temperature = log_temperature
        self.log_density = log_density
        self.log_coeff = log_coeff

        self._compute_interpolating_splines()

    @classmethod
    def from_adf11(cls, name):
        """Instantiate a RateCoefficient by reading in an adf11 file.

        Args:
            name: The /full/name/of/an/adf11 file.
        """
        adf11_data = Adf11(name).read()
        nuclear_charge = adf11_data['charge']
        element = adf11_data['element']
        filename = adf11_data['name']
        #filename is probably redundant:
        assert name == filename

        log_temperature = adf11_data['log_temperature']
        log_density = adf11_data['log_density']
        log_coeff = adf11_data['log_coeff']

        return cls(nuclear_charge, element, log_temperature, log_density,
                log_coeff, name=filename)

    def copy(self):
        log_temperature = self.log_temperature.copy()
        log_density = self.log_density.copy()
        log_coeff = self.log_coeff.copy()
        cls = self.__class__(self.nuclear_charge, self.element, log_temperature,
                log_density, log_coeff, self.adf11_file)
        return cls

    def _compute_interpolating_splines(self):
        self.splines = []
        # if we want to implement metastables there are more sets of coefficients
        # than just for nuclear charge. This should be TODO'd.
        # Also for stuff like ecd: ionisation potentials there is nuclear_charge + 1
        for k in range(self.nuclear_charge):
            x = self.log_temperature
            y = self.log_density
            z = self.log_coeff[k]
            self.splines.append(RectBivariateSpline(x, y, z))

    def __call__(self, k, Te, ne):
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
        c = self.log10(k, Te, ne)
        return np.power(10, c)

    def log10(self, k, Te, ne):
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

        Te, ne = np.broadcast_arrays(Te, ne)
        log_temperature = np.log10(Te)
        log_density = np.log10(ne)

        c = self.splines[k](log_temperature, log_density, grid=False)
        return c

    @property
    def temperature_grid(self):
        """Get a np.array of temperatures in [eV]."""
        return 10**(self.log_temperature)

    @property
    def density_grid(self):
        """Get an np.array of densities in [m^3]."""
        return 10**(self.log_density)

from sys import float_info
class ZeroCoefficient(RateCoefficient):
    """A subclass of RateCoefficient"""
    def __init__(self):
        pass

    def __call__(self, k, Te, ne):
        Te, ne = np.broadcast_arrays(Te, ne)
        return float_info.min * np.ones_like(Te)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# collision_radiative
from __future__ import absolute_import
import numpy as np

from .abundance import FractionalAbundance


class CollRadEquilibrium(object):
    def __init__(self, atomic_data):
        self.atomic_data = atomic_data
        self.ionisation_coeff = atomic_data.coeffs['ionisation'] # RateCoefficient objects
        self.recombination_coeff = atomic_data.coeffs['recombination']
        self.nuclear_charge = atomic_data.nuclear_charge #could be generalized to include metastables?

    def ionisation_stage_distribution(self, temperature, density):
        """Compute ionisation stage fractions for collrad equilibrium.

        This case only includes ionisation and recombination.
        It does not include charge exchange, or any time-dependent effects.

        Args:
            temperature (array_like): temperatures [eV].
            density (array_like): densities [m^-3].

        Returns:
            A FractionalAbundance object
        """
        if len(temperature) == 1 and len(density) > 1:
            temperature = temperature * np.ones_like(density)
        y = np.zeros((self.nuclear_charge + 1, len(temperature)))
        y[0] = np.ones_like(temperature)
        for k in range(self.nuclear_charge):
            S = self.ionisation_coeff(k, temperature, density)
            alpha = self.recombination_coeff(k, temperature, density)
            y[k+1] = y[k] * S / alpha

        y /= y.sum(0) # fractional abundance
        return FractionalAbundance(self.atomic_data, y, temperature, density)


if __name__ == '__main__':
    pass
# time_dependent_rates
from __future__ import absolute_import
import numpy as np
import scipy.integrate

from .abundance import FractionalAbundance
from .collisional_radiative import CollRadEquilibrium

class RateEquations(object):
    """
    Attributes:
        atomic_data: an AtomicData object that these equations correspond to.
        nuclear_charge: AtomicData's charge.

        -the rest of these attributes are not set until after initializion-
        temperature: a list of temperature points at which the rate equations
            will be integrated.
        density: a float represing the density.
        y: an np.array of dimensions self.y_shape.
            Used as initial conditions for the integrator.
        y_shape: (nuclear_charge+1, # temperature points)
        S: ionisation coefficients. Has dimensions of y_shape.
        alpha: recombination coefficients. Has dimensions of y_shape.
            Both have arrays of zeros at their highest index.
        dydt: array with dimensions y_shape. Initialized to zero, 
            gets changed by derivs().
    """
    def __init__(self, atomic_data):
        self.atomic_data = atomic_data
        self.nuclear_charge = atomic_data.nuclear_charge

    def _set_temperature_and_density_grid(self, temperature, density):
        self.temperature = temperature
        self.density = density

    def _set_initial_conditions(self):
        self.y_shape = (self.nuclear_charge + 1, len(self.temperature))
        self._init_y()
        self._init_coeffs()

    def _init_y(self):
        """Start with the ions all in the +0 state at t=0."""
        y = np.zeros(self.y_shape)
        y[0] = np.ones_like(self.temperature)
        self.y = y.ravel() #functions like MMA's Flatten[] here
        self.dydt = np.zeros(self.y_shape)

    def _init_coeffs(self):
        """Initialises ionisation and recombination coefficents S and alpha.
        
        Note that S[-1] and alpha[-1] will both be arrays of zeros.
        (With length of self.temperature)
        """
        S = np.zeros(self.y_shape)
        alpha = np.zeros(self.y_shape)

        recombination_coeff = self.atomic_data.coeffs['recombination']
        ionisation_coeff = self.atomic_data.coeffs['ionisation']
        for k in range(self.nuclear_charge):
            S[k] = ionisation_coeff(k, self.temperature, self.density)
            alpha[k] = recombination_coeff(k, self.temperature, self.density)

        self.S = S
        self.alpha = alpha

    def derivs(self, y_, t0):
        """Construct the r.h.s. of the rate equations.

        This function is executed several times for each odeint step.
        If 100 'times' happen this was executed probably 500 times?
        Also since dydt is an array, the assignment is a copy so
        it gets changed each call.
        """
        dydt = self.dydt
        S = self.S
        alpha_to = self.alpha
        ne = self.density

        # shape the y into a 2D array so that it's easier to apply
        # rate coeffients to specific charge states, then switch it back
        # to a 1D array so that odeint can handle it: y0 must be 1D.
        y = y_.reshape(self.y_shape)
        current = slice(1, -1) # everything but the first and last
        upper = slice(2, None)
        lower = slice(None, -2)
        dydt[current]  = y[lower] * S[lower]
        dydt[current] += y[upper] * alpha_to[current]
        dydt[current] -= y[current] * S[current]
        dydt[current] -= y[current] * alpha_to[lower]

        current, upper = 0, 1 # neutral and single ionised state
        dydt[current] = y[upper] * alpha_to[current] - y[current] * S[current]

        current, lower = -1, -2 # fully stripped and 1 electron state
        dydt[current] = y[lower] * S[lower] - y[current] * alpha_to[lower]
        dydt *= ne

        return dydt.ravel()

    def solve(self, time, temperature, density):
        """
        Integrate the rate equations.

        Args:
            time (np.array): A sequence of time points for which to solve.
            temperature (np.array): Electron temperature grid to solve on [eV].
            density (float): Electron density grid to solve on [m^-3].

        Returns:
            a RateEquationSolution
        """
        self._set_temperature_and_density_grid(temperature, density)
        self._set_initial_conditions()
        solution  = scipy.integrate.odeint(self.derivs, self.y, time)

        abundances = []
        for s in solution.reshape(time.shape + self.y_shape):
            abundances.append(FractionalAbundance(self.atomic_data, s, self.temperature,
                self.density))

        return RateEquationsSolution(time, abundances)


class RateEquationsWithDiffusion(RateEquations):
    """ Represents a solution of the rate equations with diffusion
    out of every charge state with a time constant of tau, 
    and fueling of neutrals so that the population is constant in time.
    """
    def derivs(self, y, t):
        dydt = super(self.__class__, self).derivs(y, t)

        tau = self.diffusion_time

        dydt -= y/tau
        dydt = dydt.reshape(self.y_shape)
        dydt[0] += 1/tau # ensures stable population of 1
        return dydt.ravel()

    def solve(self, time, temperature, density, diffusion_time):
        self.diffusion_time = diffusion_time
        return super(self.__class__, self).solve(time, temperature, density)


class RateEquationsSolution(object):
    def __init__(self, times, abundances):
        self.times = times
        self.abundances = abundances

        self._find_parameters()
        self._compute_y_in_collrad()

    def _find_parameters(self):
        y = self.abundances[0]
        self.atomic_data = y.atomic_data
        self.temperature = y.temperature
        self.density = y.density

    def _compute_y_in_collrad(self):
        """
        Compute the corresponding ionisation stage distribution in collrad
        equilibrum.
        """
        eq = CollRadEquilibrium(self.atomic_data)
        y_collrad = eq.ionisation_stage_distribution(self.temperature,
                self.density)

        self.y_collrad = y_collrad

    # I guess this is halfway to beng an Immutable Container?
    def __getitem__(self, key):
        if not isinstance(key, int) and not isinstance(key, np.integer):
            raise TypeError('key must be integer.')
        return self.abundances[key]

    def at_temperature(self, temperature_value):
        """Finds the time evolution of ionisation states at fixed temperature.
        
        The temperature is the next one in self.temperature
            that is larger than temperature_value.
        """
        temperature_index = np.searchsorted(self.temperature,
                temperature_value)

        return np.array([y.y[:, temperature_index] for y in self.abundances])

    def mean_charge(self):
        """Returns an array with shape (len(times), len(temperature))."""
        return np.array([f.mean_charge() for f in self.abundances])

    def steady_state_time(self, rtol=0.01):
        """Find the times at which the ys approach steady-state.
        
        Returns: an array of steady-state time for each temperature.
        E.g. if it took a 1 eV plasma 1.0s and a 10 eV plasma 0.2s,
        np.array([1.0, 0.2])
        """
        z_mean_ref = self.y_collrad.mean_charge()

        tau_ss = np.zeros_like(self.temperature)
        for t, f in reversed(list(zip(self.times, self.abundances))):
            z_mean = f.mean_charge()
            mask = np.abs(z_mean/z_mean_ref - 1) <= rtol
            tau_ss[mask] = t

        return tau_ss

    def ensemble_average(self):

        tau = self.times[:, np.newaxis, np.newaxis]
        y = [y.y for y in self.abundances]
        # y is a list of np.arrays. len(y) is nTimes, and
        # the contained arrays have shape (nuclear_charge + 1, nTemperatures)
        y_bar = scipy.integrate.cumtrapz(y, x=tau, axis=0)
        y_bar /= tau[1:]
        # y_bar is a 3D np.array with shape
        # (nTimes - 1, nuclear_charge + 1, nTemperatures)
        return self._new_from(tau.squeeze(), y_bar)

    def select_times(self, time_instances):
        indices = np.searchsorted(self.times, time_instances)
        f = [self[i] for i in indices]
        times = self.times[indices]

        return self.__class__(times, f)

    # is this implementing the iterator protocol?
    # https://www.ibm.com/developerworks/library/l-pycon/ ??
    def __iter__(self):
        for y in self.abundances:
            yield y

    def _new_from(self, times, concentrations):
        new_concentrations = []
        for y in concentrations:
            f = FractionalAbundance(self.atomic_data, y, self.temperature,
                    self.density)
            new_concentrations.append(f)
        return self.__class__(times, new_concentrations)


if __name__ == '__main__':
    pass

# radiation
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from .atomic_data import ZeroCoefficient
from functools import reduce


class Radiation(object):
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
    def __init__(self, ionisation_stage_distribution, impurity_fraction=1.,
            neutral_fraction=0.):
        self.y = ionisation_stage_distribution
        self.atomic_data = ionisation_stage_distribution.atomic_data

        self.temperature = self.y.temperature
        self.electron_density = self.y.density

        self.impurity_fraction = impurity_fraction
        self.neutral_fraction = neutral_fraction

    @property
    def power(self):
        return self._compute_power()

    @property
    def specific_power(self):
        """Power per electron per impurity nucleus, [W m^3]"""
        power = self.power
        for key in power.keys():
            power[key] /= self.electron_density * self.get_impurity_density()
        return power

    def get_impurity_density(self):
        return self.impurity_fraction * self.electron_density

    def get_neutral_density(self):
        return self.neutral_fraction * self.electron_density

    def _get_power_coeffs(self):
        """Get a dict of RateCoefficient objects.
        Looks for RateCoefficients called
        line_power, continuum_power, and cx_power.
        If one is not found, it returns a ZeroCoefficient.

        Returns:
            {'continuum_power': <RateCoefficient object>,
             'line_power'     : <RateCoefficient object>,
             'cx_power'       : <ZeroCoefficient object>}
        """
        power_coeffs = {}
        for key in ['line_power', 'continuum_power', 'cx_power']:
            power_coeffs[key] = self.atomic_data.coeffs.get(key,
                    ZeroCoefficient())
        return power_coeffs

    def _compute_power(self):
        """
        Compute radiation power density in [W/m3].
        """
        shape_ = self.atomic_data.nuclear_charge, self.temperature.shape[0]

        power_coeffs = self._get_power_coeffs()
        radiation_power = {}
        for key in power_coeffs.keys():
            radiation_power[key] = np.zeros(shape_)

        ne = self.electron_density
        ni = self.get_impurity_density()
        n0 = self.get_neutral_density()
        y = self.y

        for k in range(self.atomic_data.nuclear_charge):
            for key in radiation_power.keys():
                coeff = power_coeffs[key](k, self.temperature,
                        self.electron_density)

                if key in ['continuum_power', 'line_power']:
                    if key in ['continuum_power']:
                        scale = ne * ni * y.y[k + 1]
                    else:
                        scale = ne * ni * y.y[k]
                elif key in ['cx_power']:
                    scale = n0 * ni * y.y[k]

                radiation_power[key][k] = scale * coeff

        # compute the total power
        radiation_power['total'] = reduce(lambda x,y: x+y,
                radiation_power.values())

        # sum over all ionisation stages
        for key in radiation_power.keys():
            radiation_power[key] = radiation_power[key].sum(0)

        return radiation_power

    def plot(self, **kwargs):
        """Plot the specific power for line_power, continuum_power,
        cx_power, and the total.

        Possible kwargs:
            'x': the x values of temperature for the plot.
                This will make the xscale linear.
            'ax': something about the axes; I don't understand yet.
        """
        if 'x' in kwargs:
            xscale = 'linear'
        else:
            xscale = 'log'

        ax = kwargs.get('ax', plt.gca()) # gca is get current axes
        x = kwargs.get('x', self.temperature)

        lines = []
        for key in ['line_power', 'continuum_power', 'cx_power', 'total']:
            p = self.specific_power[key]
            l, = ax.semilogy(x, p, label=self._get_label(key))
            lines.append(l)

        ax.set_xscale(xscale)
        ax.set_xlabel(r'$T_\mathrm{e}\ [\mathrm{eV}]$')
        ax.set_ylabel(r'$P\ [\mathrm{W/m^3}]$')

        self._decorate_plot(ax, lines)
        plt.draw_if_interactive()

        return lines

    def _decorate_plot(self, ax, lines):
        """called by plot()
        Args:
            ax: the axes
            lines: different ax.semilogy (plots I suppose)?
        """
        alpha = 0.5 # transparency for fancy filling
        min_ = ax.get_ylim()[0]
        baseline = min_ * np.ones_like(self.temperature)

        for line in lines[:-1]:
            x = line.get_xdata()
            y = line.get_ydata()
            ax.fill_between(x, y, baseline, color=line.get_color(), alpha=alpha)
        lines[-1].set_color('black')

    def _get_label(self, key):
        """Called by plot"""
        labels = {
            'continuum_power' : 'continuum',
            'line_power' : 'line',
            'cx_power' : 'charge-exchange',
            'total' : 'total',
        }
        return labels.get(key, None)

# electron_cooling
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from .atomic_data import ZeroCoefficient
from .radiation import Radiation
from functools import reduce

# This class is almost a copy of Radiation; only 
# _get_staging_coeffs / _get_power_coeffs and _compute_power are different.
class ElectronCooling(object):
    """
    Electron cooling power is radiation + ionisation - recombination.

    This class calculates electron cooling power for an ionisation 
    stage distribution at given electron densities and temperatures.

    In collisional radiative equilibrium, ElectronCooling.power will
    equal Radiation.power, since the same amount of ions are being 
    ionised and recombined each second.

    Attributes:
        rad: a Radiation object.
        y: a FractionalAbundance object.
        atomic_data: that FrAb's atomic data.
        temperature (np.array): that FrAb's temperature list.
        electron_density (np.array or float): that FrAb's density list.
        impurity_fraction: ???
            n_impurity = n_e * impurity_fraction
        neutral_fraction: fraction of neutral hydrogen, for cx_power.
            n_n = n_e * neutral_fraction
    """
    def __init__(self, ionisation_stage_distribution, impurity_fraction=1.,
            neutral_fraction=0.):
        self.y = ionisation_stage_distribution
        self.atomic_data = ionisation_stage_distribution.atomic_data

        self.temperature = self.y.temperature
        self.electron_density = self.y.density

        self.impurity_fraction = impurity_fraction
        self.neutral_fraction = neutral_fraction

        self.eV = 1.6e-19

        self.rad = Radiation(ionisation_stage_distribution, impurity_fraction,
            neutral_fraction)


    @property
    def power(self):
        return self._compute_power()

    @property
    def specific_power(self):
        """Power per electron per impurity nucleus, [W m^3]"""
        power = self.power
        for key in power.keys():
            power[key] /= self.electron_density * self.get_impurity_density()
        return power

    def epsilon(self, tau):
        """Electron cooling energy per ion [eV], given the lifetime tau."""
        eps = {}
        for k in 'rad_total', 'total':
            eps[k] = self.specific_power[k] * self.electron_density * tau / self.eV
        return eps

    def get_impurity_density(self):
        return self.impurity_fraction * self.electron_density

    def get_neutral_density(self):
        return self.neutral_fraction * self.electron_density

    def _get_staging_coeffs(self):
        """Get a dict of RateCoefficient objects.

        Returns:
            {'ionisation':          : <RateCoefficient object>,
             'recombination'        : <RateCoefficient object>,
             'ionisation_potential' : <RateCoefficient object>}
        """
        staging_coeffs = {}
        for key in ['ionisation', 'recombination', 'ionisation_potential']:
            staging_coeffs[key] = self.atomic_data.coeffs.get(key,
                    ZeroCoefficient())
        return staging_coeffs

    def _compute_power(self):
        """
        Compute electron cooling power density in [W/m3].
        """

        shape_ = self.atomic_data.nuclear_charge, self.temperature.shape[0]

        staging_coeffs = self._get_staging_coeffs()
        staging_power = {}
        staging_power_keys = 'ionisation', 'recombination'
        for key in staging_power_keys:
            staging_power[key] = np.zeros(shape_)

        ne = self.electron_density
        ni = self.get_impurity_density()
        y = self.y # a FractionalAbundance

        for k in range(self.atomic_data.nuclear_charge):
            # in joules per ionisation stage transition
            # note that the temperature and density don't matter for the potential.
            potential = self.eV * staging_coeffs['ionisation_potential'](k, self.temperature, self.electron_density)
            for key in staging_power_keys:
                coeff = staging_coeffs[key](k, self.temperature,
                        self.electron_density)

                # somewhat ugly...
                if key == 'recombination':
                    sign, shift = -1, 1
                else:
                    sign, shift = 1, 0

                scale = ne * ni * y.y[k+shift] * potential

                staging_power[key][k] = sign * scale * coeff

        # sum over all ionisation stages
        for key in staging_power.keys():
            staging_power[key] = staging_power[key].sum(0)

        # now get the radiation power.
        # gets a dict with keys line, continuum, cx, total
        rad_power = self.rad._compute_power() 
        # save rad_total for later
        rad_total = rad_power.pop('total')

        # Comment from previous dev
        # this is a Bad Idea on how to merge two dicts but oh well
        # http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
        # Previous code
        # cooling_power = dict(rad_power.items() + staging_power.items())

        # First sure keys are unique
        # If in debugging mode only for speed
        if __debug__:
            keys_rad = set(rad_power.keys())
            keys_staging = set(staging_power.keys())
            intersection = keys_rad & keys_staging
            assert len(intersection) is 0

        cooling_power = {**rad_power, **staging_power}
        cooling_power['total'] = reduce(lambda x,y: x+y,
                cooling_power.values())
        cooling_power['rad_total'] = rad_total

        return cooling_power
