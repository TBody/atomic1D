# Program name: atomic1D/build_json.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 14 July 2017
# 
# 
# Copies a file called 'data_dict' into a .json file specified as 'filename'
# 
# Can call from an interactive shell with
# >> exec(open('JSON_export.py').read())


from copy import deepcopy
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