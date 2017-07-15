# Program name: atomic1D/Prad.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 15 July 2017
# 
# Program function: output the radiated power (Prad) and electron cooling power (Pcool)
#                   by using OpenADAS rates on output JSON from SD1D run
# 
# Under active development: <<TODO>> indicates development goal

from atomic1D import ImpuritySpecies #Load the impurity species class
from atomic1D import sharedFunctions

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

def processCommandLineArguments():
	# Processes the command line arguments supplied to Prad.py, to set paths to input_file and JSON_database_path
	# 
	# input: none (all supplied via command line)
	# return:   input_file -> JSON file from SD1D run
	#           JSON_database_path -> 
	import sys, os

	# Set defaults
	input_file = '' #Path to output JSON file from SD1D
	JSON_database_path = 'json_database' #Path to json_database, which must have a subdirectory json_data with JSON files from OpenADAS
	element = '' #Impurity species being studied

	# Check command line arguments
	for command_line_argument in sys.argv:
		if 'Prad.py' in command_line_argument:
			# First argument will always be the function-name - skip this
			continue
		elif '-help' == command_line_argument:
			print('Function Prad.py called with -help')
			print('Function: calculate the radiated power along a field line')
			print('Inputs: -i=input_file')
			print('        -> path to a JSON file generated by data_dict_export.py')
			print('           operating in a SD1D I/O folder (i.e. case-##)')
			print('        -z=impurity element')
			print('        -> impurity species for which to calculate the radiative loss')
			print('        -jpath=JSON_database_path')
			print('        -> path to a folder which contains a subdirectory json_data.')
			print('           json_data should contain JSON files corresponding to OpenADAS')
			print('           .dat files. Generated by running make json_update on the')
			print('           makefile of TBody/atomic1D')
			quit()
		elif '-i' == command_line_argument[0:2]:
			# -i indicates input file 
			input_file = command_line_argument[3:]
		elif '-z' == command_line_argument[0:2]:
			# -element indicates specification of impurity element
			# Current supported are 'C'/'Carbon' and 'N'/'Nitrogen'
			element = command_line_argument[3:]
		elif '-jpath' == command_line_argument[0:6]:
			# -jpath indicates path to JSON database
			JSON_database_path = command_line_argument[7:]
		else:
			raise RuntimeError('Command ({}) not recognised'.format(command_line_argument))

	if input_file == '':
		input_file = input('Path to SD1D-output JSON file: ')

	if element == '':
		element = input('Impurity element: ')

	if os.path.isfile(input_file):
		print("Input file: {}".format(input_file))
	else:
		raise FileNotFoundError("Input file ({}) not found".format(input_file))

	if os.path.isdir(JSON_database_path):
		if os.path.isdir(JSON_database_path+'/json_data'):
			print("OpenADAS JSON database: {}".format(JSON_database_path))
		else:
			raise RuntimeError("Subdirectory json_data not found in JSON database. Might need to run make json_update on atomic1D")
	else:
		raise FileNotFoundError("OpenADAS JSON database ({}) not found".format(JSON_database_path))

	e = element.lower()

	try:
		impurity = ImpuritySpecies(e)
	except KeyError:
		raise NotImplementedError("Impurity element ({}) not yet implemented".format(e))
	print('Element: {}, year: {}, has cx power: {}'.format(impurity.name,impurity.year,impurity.has_charge_exchange))

	return [input_file, JSON_database_path, impurity]
	
def processInputFile(input_file):
	# process a input JSON file to extract Te(s,t), ne(s,t), ne/nn (s,t)
	# n.b. s refers to the upstream distance from the strike-point
	#      t is time (will need to add normalisation factor <<TODO>> to convert to real-time)
	# 
	# input:    input_file -> JSON file from SD1D run
	# return:   Te, ne, neutral_fraction

	# input_file can be either relative or absolute path to JSON file
	# from atomic1D import retrieveFromJSON

	data_dict = sharedFunctions.retrieveFromJSON(input_file)

	# Retrieve (normalised values)
	Ne = data_dict['Ne']
	Nn = data_dict['Nn']
	# P = 2*Ne*Te => Te = P/(2*Ne)
	# N.b. division between two numpy ndarrays is piecewise
	T  = data_dict['P']/(2*data_dict['Ne'])
	
	# Neutral fraction affects charge exchange
	neutral_fraction = Nn/Ne

	# Dimensions are [t, x, y, z]
	#                [0, 1, 2, 3]
	# SD1D only outputs on time (t) and distance from strike-point (stored as y)
	# Make sure these dimensions are len == 1, in case later version of SD1D changes which index distance is stored in
	assert Ne.shape[1] == 1 #Before you panic, remember Python indexes from 0
	assert Ne.shape[3] == 1
	# Remove len == 1 dimensions
	Ne               = Ne[: ,0 ,: ,0]
	neutral_fraction = neutral_fraction[: ,0 ,: ,0]
	T                = T[: ,0 ,: ,0]
	
	# Retrieve normalisation factors
	Nnorm = data_dict['Nnorm']
	Tnorm = data_dict['Tnorm']
	# Converts N into m^-3, T into eV

	return [T*Tnorm, Ne*Nnorm, neutral_fraction]

def calculateCollRadEquilibrium(impurity, temperature, density, t):
	# Calculates the fractional distribution across ionisation stages, assuming generalised collisional-radiative
	# equilibrium (i.e. effective ionisation and recombination rates are the only significant terms, charge-exchange
	# is ignored)
	# 
	# Inputs: 	impurity 		= ImpuritySpecies object (the impurity for which we are calculating the radiation)
	# 			density    		= electron/ion density (in m^-3)
	# 			temperature 	= electron/ion density (in eV)
	# 
	import numpy as np

	# Will use the atomic number a lot in this function - extract it for easy use
	Z = impurity.atomic_number

	# Calculating for 1D linked data, output from SD1D
	data_length = len(temperature[t, :])
	# Assumed that temperature and density data are of the same length - check this before continuing
	assert data_length == len(density[t, :])

	# Preallocate an empty array for
	y = np.zeros((Z + 1, data_length))

	# Set the ground state density to zero (arbitrary - will normalise later)
	y[0,:] = 1

	for k in range(Z): #will return k = 0, 1, ..., Z-1
		iz_coeffs = impurity.rate_coefficients['ionisation'].call1D(k, temperature[t,:], density[t,:])
		recc_coeffs = impurity.rate_coefficients['recombination'].call1D(k, temperature[t,:], density[t,:])

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

	

if __name__ == '__main__':

	# Process command line arguments to set the path to the input file (from SD1D)
	# and the JSON database (from make json_update)
	[input_file, JSON_database_path, impurity] = processCommandLineArguments()
	
	# Add the JSON files associated with this impurity to its .adas_files_dict attribute
	# where the key is the (extended) process name, which maps to a filename (string)
	# Check that these files exist in the JSON_database_path/json_data/ directory
	for key, value in datatype_abbrevs.items():
		if impurity.has_charge_exchange or not(value in {'ccd', 'prc'}):
			impurity.addJSONFiles(key,value,JSON_database_path)

	# Use the .adas_file_dict files to generate RateCoefficient objects for each process
	# Uses the same keys as .adas_file_dict
	impurity.makeRateCoefficients(JSON_database_path)
	
	# Inspections on the RateCoefficient object
	# print(impurity.rate_coefficients['ionisation'])	
	# impurity.rate_coefficients['ionisation'].inspect_with_plot(1)
	# impurity.rate_coefficients['ionisation'].inspect_interp_with_plot(1)

	# Process the input_file to extract
	# 	density(t,s) 					= electron density (in m^-3)
	# 	temperature(t,s)				= electron/ion temperature (in eV)
	# 	neutral_fraction(t,s)			= neutral density/electron density (no units)
	# where t is time index, s is 1D distance index
	[temperature, density, neutral_fraction] = processInputFile(input_file)

	# First write the code for a single time-step
	# Then extend to calculate for all time-steps <<TODO>>
	t = 0

	# Calculate the distribution across ionisation stages, assuming collisional-radiative equilibrium
	iz_stage_distribution = calculateCollRadEquilibrium(impurity, temperature, density, t)

	# compute power





























