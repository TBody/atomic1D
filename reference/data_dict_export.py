# Program name: atomic1D/build_json.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 14 July 2017
# 
# 
# Makes data_dict and copies it into a .json file 'sd1d-case-05.json'

filename = 'sd1d-case-05'

from boutdata.collect import collect

data_dict = {}

# Normalisation factor for temperature - T * Tnorm returns in eV
data_dict["Tnorm"] = collect("Tnorm")
# Normalisation factor for density - N * Nnorm returns in m^-3
data_dict["Nnorm"] = collect("Nnorm")
# Plasma pressure (normalised). Pe = 2 Ne Te => P/Ne = Te (and assume Ti=Te)
data_dict["P"] = collect("P")
# Electron density (normalised)
data_dict["Ne"] = collect("Ne")
# Neutral density (normalised)
data_dict["Nn"] = collect("Nn")

# Help for user
data_dict["Help"] = "Contains outputs from Boutprojects/SD1D/case-05 example. Created with data_dict_export.py - stored in Github.com/TBody/atomic1D/reference"

from copy import deepcopy
import numpy as np
import json

# Need to 'jsonify' the numpy arrays (i.e. convert to nested lists) so that they can be stored in plain-text
# Deep-copy data to a new dictionary and then edit that one (i.e. break the data pointer association - keep data_dict unchanged in case you want to run a copy-verify on it)

data_dict_jsonified = deepcopy(data_dict)

numpy_ndarrays = [];
for key, element in data_dict.items():
    if type(element) == np.ndarray:
        # Store which keys correspond to numpy.ndarray, so that you can de-jsonify the arrays when reading
        numpy_ndarrays.append(key)
        data_dict_jsonified[key] = data_dict_jsonified[key].tolist()

# Encode help
# >> data_dict['help'] = 'help string'

# <<Use original filename, except with .json instead of .dat extension>>
with open('{}.json'.format(filename),'w') as fp:
    json.dump(data_dict_jsonified, fp, sort_keys=True, indent=4)