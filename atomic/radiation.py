from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from .atomic_data import ZeroCoefficient
from functools import reduce
import pdb

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
        # pdb.set_trace()
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

