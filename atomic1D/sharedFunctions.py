
def retrieveFromJSON(file_name):
	# Inputs - a JSON file corresponding to an OpenADAS .dat file or SD1D output file
	# file_name can be either relative or absolute path to JSON file
	# Must have .json extension and match keys of creation
	# Not need for the .dat -> .json conversion, but included for reference
	import json
	from warnings import warn
	from copy import deepcopy
	import numpy as np

	file_extension  = file_name.split('.')[-1] #Look at the extension only (last element of split on '.')
	if file_extension != 'json':
		raise NotImplementedError('File extension (.{}) is not .json'.format(file_extension))

	with open(file_name,'r') as fp:
		data_dict = json.load(fp)

	if  set(data_dict.keys()) != {'numpy_ndarrays', 'charge', 'help', 'log_density', 'number_of_charge_states', 'log_temperature', 'element', 'log_coeff', 'name', 'class'}\
	and set(data_dict.keys()) != {'numpy_ndarrays', 'help', 'Ne', 'Tnorm', 'P', 'Nnorm', 'Nn'}:
		warn('Imported JSON file {} does not have the expected set of keys - could result in an error'.format(file_name))

	# Convert jsonified numpy.ndarrays back from nested lists
	data_dict_dejsonified = deepcopy(data_dict)

	for key in data_dict['numpy_ndarrays']:
		data_dict_dejsonified[key] = np.array(data_dict_dejsonified[key])

	return data_dict_dejsonified