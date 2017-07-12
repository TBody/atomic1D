# Program name: atomic1D/build_json.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 12 July 2017

import json
import pickle

# for pickling/unpickling
# https://www.thoughtco.com/using-pickle-to-save-objects-2813661
# n.b. need to use 'rb' and 'wb' to read/write bytes instead of strings
# (i.e. binary in/out)
with open('pickled_scd96_c.obj','rb') as fp:
	ret = pickle.load(fp)

iz0, is1min, is1max, nptnl, nptn, nptnc, iptnla, iptna, iptnca, ncnct,\
icnctv, iblmx, ismax, dnr_ele, dnr_ams, isppr, ispbr, isstgr, idmax,\
itmax, ddens, dtev, drcof, lres, lstan, lptn = ret

data = {}
data['charge'] = iz0
data['log_density'] = ddens[:idmax]
data['log_temperature'] = dtev[:itmax]
data['number_of_charge_states'] = ismax
data['log_coeff'] = drcof[:ismax, :itmax, :idmax]

data['class'] = 'scd'
data['element'] = 'c'
data['name'] = '/Users/thomasbody/Dropbox/Thesis/atomic/adas_data/scd96_c.dat'

# convert everything to SI + eV units
data['log_density'] += 6 # log(cm^-3) = log(10^6 m^-3) = 6 + log(m^-3)
# the ecd (ionisation potential) class is already in eV units.
# admittedly this a cluge to store these non-rate-coefficient
# objects inside a RateCoefficient but it'll save a bunchton of code.
if data['class'] != 'ecd':
    data['log_coeff'] -= 6 # log(m^3/s) = log(10^-6 m^3/s) = -6 + log(m^3/s)
else:
    data['log_coeff'] = np.log10(data['log_coeff'][1:])

# On how to jsonify numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# to jsonify
	# import numpy as np
	# import codecs, json 
	# a = np.arange(10).reshape(2,5) # a 2 by 5 array
	# b = a.tolist() # nested lists with same data, indices
	# file_path = "/path.json" ## your path variable
	# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
# to unjsonify
	# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
	# b_new = json.loads(obj_text)
	# a_new = np.array(b_new)

# Print out the file type for each element in the data
for key, element in data.items():
	print("data key: {:30} data type: {}".format(key,type(element)))
# Find that log_density, log_temperature, log_coeff are numpy.ndarry

# jsonify the numpy arrays
import numpy as np
# copy data to a new dictionary (data_jsonified) and then edit that one
from copy import deepcopy
data_jsonified = deepcopy(data)
data_jsonified['log_density']     = data_jsonified['log_density'].tolist()
data_jsonified['log_temperature'] = data_jsonified['log_temperature'].tolist()
data_jsonified['log_coeff']       = data_jsonified['log_coeff'].tolist()

with open('data.json','w') as fp:
	json.dump(data_jsonified, fp, sort_keys=True, indent=4)

# using codecs doesn't actually significantly change the output - don't bother
# import codecs
# json.dump(data, codecs.open('data.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

# data_jsonified = None

print("\nCleared data, then reloading from JSON\n")

with open('data.json','r') as fp:
	data3 = json.load(fp)

data3['log_density']     = np.array(data3['log_density'])
data3['log_temperature'] = np.array(data3['log_temperature'])
data3['log_coeff']       = np.array(data3['log_coeff'])

for key, element in data.items():
	if type(element) != np.ndarray:
		print("data key: {:30} data type: {:30} Copy successful: {}".format(key, str(type(element)), data[key] == data3[key]))
	else:
		print("data key: {:30} data type: {:30} Copy successful: {}".format(key, str(type(element)), np.array_equal(data[key],data3[key])))

