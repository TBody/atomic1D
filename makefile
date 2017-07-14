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