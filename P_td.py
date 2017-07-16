# Program name: atomic1D/P_td.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 16 July 2017
# 
# Program function: calculate the time-dependent rate-equations corresponding to
# 					the electron energy balance equation and the ionisation-stage
# 					population equations
#                   by using OpenADAS rates on output JSON from SD1D run
# 
# Under active development: <<TODO>> indicates development goal

from atomic1D import sharedFunctions
from atomic1D import ImpuritySpecies #Load the impurity species class
from atomic1D import SD1DData
import numpy as np

# Mapping between processes and their ADAS codes
# Note that the code will try to generate a file for each process listed
# Will return an error if file not found, except for charge exchange (ccd and prc)
# which is skipped if .has_charge_exchange = False
datatype_abbrevs = {
        'ionisation'           : 'scd',
        'recombination'        : 'acd',
        'cx_recc'              : 'ccd',
        'continuum_power'      : 'prb',
        'line_power'           : 'plt',
        'cx_power'             : 'prc',
        'ionisation_potential' : 'ecd', #N.b. ionisation_potential is not a rate-coefficient, but most of the 
                                        #methods are transferable 
}

# Invert the mapping of datatype_abbrevs
inv_datatype_abbrevs = {v: k for k, v in datatype_abbrevs.items()}

def calculateCollRadEquilibrium(impurity, experiment):
	# Calculates the fractional distribution across ionisation stages, assuming generalised collisional-radiative
	# equilibrium (i.e. effective ionisation and recombination rates are the only significant terms, charge-exchange
	# is ignored)
	# 
	# Inputs: 	impurity 				= ImpuritySpecies object (the impurity for which we are calculating the radiation)
	# 			experiment.density    	= SD1DData object with electron/ion density (in m^-3)
	# 			experiment.temperature 	= SD1DData object with electron/ion density (in eV)
	# 
	
	Z = impurity.atomic_number

	# Calculating for 1D linked data, output from SD1D
	data_length = len(experiment.temperature)
	# Assumed that temperature and density data are of the same length - check this before continuing
	assert data_length == len(experiment.density)

	# Preallocate an empty array for
	y = np.zeros((Z + 1, data_length))

	# Set the ground state density to zero (arbitrary - will normalise later)
	y[0,:] = 1

	for k in range(Z): #will return k = 0, 1, ..., Z-1 where Z = impurity.atomic_number
	                                        #i.e. iterate over all charge states except bare nucleus
		iz_coeffs = impurity.rate_coefficients['ionisation'].call1D(k, experiment.temperature, experiment.density)
		recc_coeffs = impurity.rate_coefficients['recombination'].call1D(k, experiment.temperature, experiment.density)

		# The ratio of ionisation from the (k)th stage and recombination from the (k+1)th
		# sets the equilibrium densities of the (k+1)th stage in terms of the (k)th (since
		# R = n_z * n_e * rate_coefficient)
		# N.b. Since there is no ionisation from the bare nucleus, and no recombination onto the
		# neutral (ignoring anion formation) the 'k' value of ionisation coeffs is shifted down 
		# by one relative to the recombination coeffs - therefore this evaluation actually gives the
		# balance

		y[k+1,:] = y[k,:] * (iz_coeffs/recc_coeffs)


	# Normalise such that the sum over all ionisation stages is '1' at all points
	y = y / y.sum(axis=0)
	# Make sure that the fractional abundance returned has the same shape as the input arrays
	# assert density.shape == y.shape #For [t,s] case
	assert data_length == len(y[0,:]) #For [t=0, s] case
	assert np.allclose(y.sum(axis=0), 1.0)

	return y

def plot_iz_stage_distribution(experiment, iz_stage_distribution):
	# plot the plasma temperature, density and ionisation-stage distribution as a function of position
	# For a single time step
	
	import matplotlib.pyplot as plt

	# Create iterator for distance axis
	s = range(experiment.data_shape[1])

	plt.plot(s, experiment.temperature/max(experiment.temperature),\
		'--',label='T/{:.2f}[eV]'.format(max(experiment.temperature)))

	plt.plot(s, experiment.density/max(experiment.density),\
		'--',label=r'$n_e$/{:.2e}[$m^{{-3}}$]'.format(max(experiment.density)))

	for k in range(iz_stage_distribution.shape[0]): #iterate over all charge states
		plt.plot(iz_stage_distribution[k,:],label='{}'.format(k))

	plt.xlabel('Distance (downstream, a.u.)')
	plt.ylabel('Fraction')
	plt.legend()

	plt.show()

def computeRadiatedPower(impurity, experiment, iz_stage_distribution):
	# Calculate the power radiated from each location
	# 
	

	radiated_power = {}
	stage_integrated_power = {}

	# Select the processes which provide a 'L' coefficient that contributes to radiation
	radiative_processes = ['continuum_power','line_power']
	if impurity.has_charge_exchange:
		# Include charge exchange if the impurity has this attribute
		radiative_processes.append('cx_power')

	for physics_process in radiative_processes:
		# Find the coefficient object (i.e. L s.t. Prad = L n_1 n_2)
		coeff = impurity.rate_coefficients[physics_process] #will return a RateCoefficient object

		coeff_evaluated = np.zeros((impurity.atomic_number, experiment.data_shape))
		power_evaluated = np.zeros((impurity.atomic_number, experiment.data_shape))
		for k in range(impurity.atomic_number):
			# Evaluate the coefficient
			# Will return a 1D array since call1D has grid=False on interpolation evaluation
			# 
			# Make a list of evaluated rate coefficients [W m^3], where the first index gives the
			# ionisation stage and the second index gives the position
			coeff_evaluated[k,:] = coeff.call1D(k, experiment.temperature, experiment.density)

			if physics_process is 'continuum_power':
				# Prad = L * n_e * n_z^(k+1)
				scale = experiment.density * (experiment.impurity_density * iz_stage_distribution[k+1])
			elif physics_process is 'line_power':
				# Prad = L * n_e * n_z^k+
				scale = experiment.density * (experiment.impurity_density * iz_stage_distribution[k])
			elif physics_process is 'cx_power':
				# Prad = L * n_0 * n_z^k+
				scale = (experiment.density * experiment.neutral_fraction) * (experiment.impurity_density * iz_stage_distribution[k])

			# Computed the power radiated (W/m^3)
			power_evaluated[k,:] = coeff_evaluated[k,:] * scale

		# Append the power radiated to the radiated_power dictionary
		# (i.e. add the full [Z * data_length] list to the dictionary)
		radiated_power[physics_process] = power_evaluated
		# Sum across all the ionisation stages to find total contribution from each
		# physics_process
		stage_integrated_power[physics_process] = np.sum(power_evaluated,axis=0)

	# Performs piece-wise add: has the same shape as the source
	# 'total' has sum over all physics_processes
	radiated_power['total'] = sum(radiated_power.values())
	stage_integrated_power['total'] = sum(stage_integrated_power.values())
	# total_power has total over all physics_processes and ionisation stages
	total_power = sum(radiated_power['total'])
	# Summing dimensions in different orders should give the same result
	assert np.allclose(total_power,stage_integrated_power['total'])
	
	# Check data shape
	if type(experiment.data_shape) == int:
		assert experiment.data_shape == len(total_power)
	elif type(experiment.data_shape) == np.ndarray:
		assert experiment.data_shape == total_power.shape
	else:
		raise NotImplementedError('Error checking data_shape match')

	# Copy the results to the experiment object
	experiment.radiated_power = radiated_power
	experiment.stage_integrated_power = stage_integrated_power
	experiment.total_power = total_power


if __name__ == '__main__':

	# Process command line arguments to set the path to the input file (from SD1D)
	# and the JSON database (from make json_update)
	[input_file, JSON_database_path, impurity] = sharedFunctions.processCommandLineArguments()
	
	# Add the JSON files associated with this impurity to its .adas_files_dict attribute
	# where the key is the (extended) process name, which maps to a filename (string)
	# Check that these files exist in the JSON_database_path/json_data/ directory
	for physics_process,filetype_code in datatype_abbrevs.items():
		if impurity.has_charge_exchange or not(filetype_code in {'ccd', 'prc'}):
			impurity.addJSONFiles(physics_process,filetype_code,JSON_database_path)

	# Use the .adas_file_dict files to generate RateCoefficient objects for each process
	# Uses the same keys as .adas_file_dict
	impurity.makeRateCoefficients(JSON_database_path)
	
	# Inspections on the RateCoefficient object
	# Can supply the process of interest (i.e. 'ionisation') and the ionisation stage (i.e. 1), the
	# function will return a 3D plot of the data used, and also the interpolation generated.
	# impurity.rate_coefficients['ionisation'].compare_interp_with_plot(1)

	# Process the input_file to extract
	# 	density(t,s) 					= electron density (in m^-3)
	# 	temperature(t,s)				= electron/ion temperature (in eV)
	# 	neutral_fraction(t,s)			= neutral density/electron density (no units)
	# where t is time index, s is 1D distance index
	experiment = SD1DData(input_file)
	
	t = experiment.data_shape[0]
	# Extract data for a single time-step
	experiment.selectSingleTime(t)

	# Set the fixed fraction for impurity concentration
	experiment.setImpurityFraction(1e-2)

	# Calculate the distribution across ionisation stages, assuming collisional-radiative equilibrium
	iz_stage_distribution = calculateCollRadEquilibrium(impurity, experiment)

	# Plot the ionisation stage distribution as a function of distance
	# plot_iz_stage_distribution(experiment, iz_stage_distribution)
	
	# Time-dependent rates (much more complicated!)
	# The atomic code has a built-in solver, while we'd probably be looking at using the BOUT solver
	# Look over calculation of differential equation rhs?

	# Export results/plot




























