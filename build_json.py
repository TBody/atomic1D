# Program name: atomic1D/build_json.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 12 July 2017

import json

# Use fortran helper functions to read .dat file into ret.
# 
# Key to inputs (from xxdata_11.pdf)
# type  | name   | description
# (i*4) | iunit  | unit to which input file is allocated
# (i*4) | iclass | class of data (numerical code) - see table below
# ----------------------------------------------------------
# use defaults (set in parameters) for everything below this
# ----------------------------------------------------------
# (i*4) | isdimd | maximum number of (sstage, parent, base)
#       |        | blocks in isonuclear master files
# (i*4) | iddimd | maximum number of dens values in 
#       |        | isonuclear master files
# (i*4) | itdimd | maximum number of temp values in 
#       |        | isonuclear master files
# (i*4) | ndptnl | maximum level of partitions
# (i*4) | ndptn  | maximum no. of partitions in one level
# (i*4) | ndptnc | maximum no. of components in a partition
# (i*4) | ndcnct | maximum number of elements in connection vector

# Supported adf11 data classes.  See src/xxdata_11/xxdata_11.for for all the
# twelve classes.
adf11_classes = {
    'acd' : 1, # recombination coefficients
    'scd' : 2, # ionisation coefficients
    'prb' : 4, # continuum radiation power
    'plt' : 8, # line radiation power
    'prc' : 5, # charge-exchange recombination radiation
    'ecd' : 12 # effective ionisation potential
}

# Some hard coded parameters to run xxdata_11.for routine.  The values have
# been take from src/xxdata_11/test.for, and should be OK for all files.
parameters = {
    'isdimd' : 200,
    'iddimd' : 40,
    'itdimd' : 50,
    'ndptnl' : 4,
    'ndptn' : 128,
    'ndptnc' : 256,
    'ndcnct' : 100
}

#<<FORTRAN HELPERS GO IN HERE>>

# Extract information from ret.
iz0, is1min, is1max, nptnl, nptn, nptnc, iptnla, iptna, iptnca, ncnct,\
icnctv, iblmx, ismax, dnr_ele, dnr_ams, isppr, ispbr, isstgr, idmax,\
itmax, ddens, dtev, drcof, lres, lstan, lptn = ret

# Key to outputs (from xxdata_11.pdf)
# type   | name       | description
# (i*4)  | iz0        | nuclear charge
# (i*4)  | is1min     | minimum ion charge + 1
#        |            | (generalised to connection vector index)
# (i*4)  | is1max     | maximum ion charge + 1
#        |            | (note excludes the bare nucleus)
#        |            | (generalised to connection vector index and excludes
#        |            | last one which always remains the bare nucleus)
# (i*4)  | nptnl      | number of partition levels in block
# (i*4)  | nptn()     | number of partitions in partition level
#        |            | 1st dim: partition level
# (i*4)  | nptnc(,)   | number of components in partition
#        |            | 1st dim: partition level
#        |            | 2nd dim: member partition in partition level
# (i*4)  | iptnla()   | partition level label (0=resolved root,1=
#        |            | unresolved root)
#        |            | 1st dim: partition level index
# (i*4)  | iptna(,)   | partition member label (labelling starts at 0)
#        |            | 1st dim: partition level index
#        |            | 2nd dim: member partition index in partition level
# (i*4)  | iptnca(,,) | component label (labelling starts at 0)
#        |            | 1st dim: partition level index
#        |            | 2nd dim: member partition index in partition level
#        |            | 3rd dim: component index of member partition
# (i*4)  | ncnct      | number of elements in connection vector
# (i*4)  | icnctv()   | connection vector of number of partitions
#        |            | of each superstage in resolved case
#        |            | including the bare nucleus
#        |            | 1st dim: connection vector index
# (i*4)  | iblmx      | number of (sstage, parent, base)
#        |            | blocks in isonuclear master file
# (i*4)  | ismax      | number of charge states
#        |            | in isonuclear master file
#        |            | (generalises to number of elements in
#        |            |  connection vector)
# (c*12) | dnr_ele    | CX donor element name for iclass = 3 or 5
#        |            | (blank if unset)
# (r*8)  | dnr_ams    | CX donor element mass for iclass = 3 or 5
#        |            | (0.0d0 if unset)
# (i*4)  | isppr()    | 1st (parent) index for each partition block
#        |            | 1st dim: index of (sstage, parent, base)
#        |            |          block in isonuclear master file
# (i*4)  | ispbr()    | 2nd (base) index for each partition block
#        |            | 1st dim: index of (sstage, parent, base)
#        |            |          block in isonuclear master file
# (i*4)  | isstgr()   | s1 for each resolved data block
#        |            | (generalises to connection vector index)
#        |            | 1st dim: index of (sstage, parent, base)
#        |            |          block in isonuclear master file
# (i*4)  | idmax      | number of dens values in
#        |            | isonuclear master files
# (i*4)  | itmax      | number of temp values in
#        |            | isonuclear master files
# (r*8)  | ddens()    | log10(electron density(cm-3)) from adf11
# (r*8)  | dtev()     | log10(electron temperature (eV) from adf11
# (r*8)  | drcof(,,)  | if(iclass <=9):
#        |            | 	log10(coll.-rad. coefft.) from
#        |            | 	isonuclear master file
#        |            | if(iclass >=10):
#        |            | 	coll.-rad. coefft. from
#        |            | 	isonuclear master file
#        |            | 1st dim: index of (sstage, parent, base)
#        |            | 		 block in isonuclear master file
#        |            | 2nd dim: electron temperature index
#        |            | 3rd dim: electron density index
# (l*4)  | lres       | = .true. => partial file
#        |            | = .false. => not partial file
# (l*4)  | lstan      | = .true. => standard file
#        |            | = .false. => not standard file
# (l*4)  | lptn       | = .true. => partition block present
#        |            | = .false. => partition block not present



# Make a new blank dictionary, data_dict
data_dict = {}
# Save data to data_dict
data_dict['charge']                  = iz0 #Nuclear charge
data_dict['log_density']             = ddens[:idmax]
data_dict['log_temperature']         = dtev[:itmax]
data_dict['number_of_charge_states'] = ismax
data_dict['log_coeff']               = drcof[:ismax, :itmax, :idmax]

# <<READ THESE ATTRIBUTES FROM FILENAME>>
# data_dict['class'] = 'scd'
# data_dict['element'] = 'c'
# data_dict['name'] = '/Users/thomasbody/Dropbox/Thesis/atomic/adas_data/scd96_c.dat'

# convert everything to SI + eV units
data_dict['log_density'] += 6 # log(cm^-3) = log(10^6 m^-3) = 6 + log(m^-3)
# N.b. the ecd (ionisation potential) class is already in eV units.
if data_dict['class'] != 'ecd':
    data_dict['log_coeff'] -= 6 # log(m^3/s) = log(10^-6 m^3/s) = -6 + log(m^3/s)
else:
    data_dict['log_coeff'] = np.log10(data_dict['log_coeff'][1:])

# jsonify the numpy arrays
import numpy as np
# copy data to a new dictionary (data_dict_jsonified) and then edit that one
from copy import deepcopy
data_dict_jsonified = deepcopy(data_dict)
data_dict_jsonified['log_density']     = data_dict_jsonified['log_density'].tolist()
data_dict_jsonified['log_temperature'] = data_dict_jsonified['log_temperature'].tolist()
data_dict_jsonified['log_coeff']       = data_dict_jsonified['log_coeff'].tolist()

# <<Use original filename, except with .json instead of .dat extension>>
with open('data_dict.json','w') as fp:
	json.dump(data_dict_jsonified, fp, sort_keys=True, indent=4)


# Full table of ADF11 types (for reference)
# 
# class index | type | GCR data content
# ------------\------\--------------------------------
# 1           | acd  | 	recombination coeffts
# 2           | scd  | 	ionisation coeffts
# 3           | ccd  | 	CX recombination coeffts
# 4           | prb  | 	recomb/brems power coeffts
# 5           | prc  | 	CX power coeffts
# 6           | qcd  | 	base meta. coupl. coeffts
# 7           | xcd  | 	parent meta. coupl. coeffts
# 8           | plt  | 	low level line power coeffts
# 9           | pls  | 	represent. line power coefft
# 10          | zcd  | 	effective charge
# 11          | ycd  | 	effective squared charge
# 12          | ecd  | 	effective ionisation potential
