# Program name: atomic1D/makefile
# Author: Thomas Body
# Author email: tajb500@york.ac.uk
# Date of creation: 12 July 2017
# 
# Makefile to automate the building and running of the atomic1D code
# 
# User control should be entirely contained within the header (i.e. section before the first <<command: dependancies>> line)
# Use:	verbose = true	-> print all warnings and errors to screen
# 				  false	-> print all warnings and errors to _log.txt file
# 		JSON_database_path to set the path to where json_database should be created and read from
# 
# To build and run the json builder code execute
# >> make json_update
# This will download .dat files from OpenADAS and then use fortran helper functions (in src) to read data.
# Data is saved as .json files in json_data (basename of file is unchanged)

JSON_database_path = json_database
verbose = false

json_update:
	@echo "Making JSON files from ADAS data files (with update)"
	@echo ""
	make fetch
	make setup
	make json
	@echo "JSON files successfully created (with update)"
	@echo ""

json:
	@echo "Making JSON files from ADAS data files"
	@echo ""
	mkdir -p $(JSON_database_path)/json_data
ifeq ($(verbose),true)
	cd $(JSON_database_path); python build_json.py
else
	cd $(JSON_database_path); python build_json.py &> build_json_log.txt
	@echo "see build_json_log.txt for build output and warnings/errors"
endif
	@echo ""
	@echo "JSON files successfully created"
	@echo ""
	
fetch:
	@echo "Fetching atomic data from OpenADAS"
	@echo ""
ifeq ($(verbose),true)
	cd $(JSON_database_path); python fetch_adas_data.py
else
	cd $(JSON_database_path); python fetch_adas_data.py &> fetch_adas_data_log.txt
	@echo "see fetch_adas_data_log.txt for build output and warnings/errors"
endif
	@echo ""
	@echo "Atomic data successfully retrieved"
	@echo ""

setup:
	@echo "Building fortran helper functions in src"
	@echo ""
ifeq ($(verbose),true)
	cd $(JSON_database_path); python setup_fortran_programs.py build_ext --inplace
else
	cd $(JSON_database_path); python setup_fortran_programs.py build_ext --inplace &> setup_fortran_programs_log.txt
	@echo "see setup_fortran_programs_log.txt for build output and warnings/errors"
endif
	@echo ""
	@echo "Fortran functions successfully built"
	@echo ""

clean:
	@echo "Cleaning atomic1D directory to fresh-install state"
	@echo "-> excluding adas_data and src tar files retrieved in fetch"
	rm -rf $(JSON_database_path)/__pycache__
	rm -rf $(JSON_database_path)/build
	rm -rf $(JSON_database_path)/src/_xxdata_11.cpython*
	rm -rf $(JSON_database_path)/src/_xxdata_15.cpython*
	rm -rf $(JSON_database_path)/src/_xxdata_11module.c
	rm -rf $(JSON_database_path)/src/_xxdata_15module.c
	rm -f  $(JSON_database_path)/setup_fortran_programs_log.txt
	@echo ""
	@echo "atomic1D directory cleaned (except fetch)"
	@echo ""

clean_refetch:
	@echo "Cleaning atomic1D directory to fresh-install state"
	@echo "-> including deleting adas_data and src functions"
	@echo ""
	rm -rf $(JSON_database_path)/adas_data
	rm -rf $(JSON_database_path)/__pycache__
	rm -rf $(JSON_database_path)/build
	rm -rf $(JSON_database_path)/src/_xxdata_11.cpython*
	rm -rf $(JSON_database_path)/src/_xxdata_15.cpython*
	rm -rf $(JSON_database_path)/src/xxdata_11
	rm -rf $(JSON_database_path)/src/xxdata_15
	rm -rf $(JSON_database_path)/src/_xxdata_11module.c
	rm -rf $(JSON_database_path)/src/_xxdata_15module.c
	rm -f  $(JSON_database_path)/src/xxdata_11.tar.gz
	rm -f  $(JSON_database_path)/src/xxdata_15.tar.gz
	rm -f  $(JSON_database_path)/fetch_adas_data_log.txt
	rm -f  $(JSON_database_path)/setup_fortran_programs_log.txt
	rm -rf $(JSON_database_path)/json_data
	rm -f  $(JSON_database_path)/build_json_log.txt
	@echo ""
	@echo "atomic1D directory cleaned"
	@echo ""