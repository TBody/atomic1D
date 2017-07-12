# Program name: atomic1D/setup_fortran_programs.py
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 12 July 2017
# 
# This program is copied from setup.py from cfe316/atomic, with a slight modification to 
# make Python3 compatible (changed .iteritems() to .items())
# 
# Builds the fortran helper functions contained in src into a python callable state (makes
# _xxdata_##module.c functions, and also creates the 'build' directory)
# 
# Note that in order to run successfully the program must be called as
# >>python setup_fortran_programs.py build_ext --inplace
# (see https://docs.python.org/3.6/distutils/configfile.html for info on setup files)
# The following files must be provided in source
#   src/helper_functions.for
#   src/xxdata_11.pyf
#   src/xxdata_15.pyf
#   

import os

extension_modules = {}
directory = 'src/xxdata_11'
sources = ['xxdata_11.for', 'xxrptn.for', 'i4unit.for',
    'i4fctn.for', 'xxword.for', 'xxcase.for', 'xfelem.for', 'xxslen.for',
     '../xxdata_11.pyf', '../helper_functions.for']
extension_modules['_xxdata_11'] = dict(sources=sources, directory=directory)

directory = 'src/xxdata_15'
sources = ['xxdata_15.for', 'xxrptn.for', 'xxmkrp.for', 'i4unit.for',
    'i4fctn.for', 'r8fctn.for', 'xxhkey.for', 'xxword.for', 'xxcase.for',
    'i4eiz0.for', 'xfelem.for', 'xxslen.for',
     '../xxdata_15.pyf', '../helper_functions.for']
extension_modules['_xxdata_15'] = dict(sources=sources, directory=directory)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('atomic', parent_package, top_path)

    for module, values in extension_modules.items():
        directory = values['directory']
        sources = values['sources']
        sources = [os.path.join(directory, i) for i in sources]

        config.add_extension(module, sources)
    return config

if __name__ == '__main__':
    print('>> setup_fortran_programs.py called')
    print('\nBuilding fortran helper functions in src\n')

    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
    print('\n>> setup_fortran_programs.py exited')

