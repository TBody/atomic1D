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

	@echo ""
	@echo "JSON files successfully created"
	@echo ""
	

fetch: fetch_adas_data.py
	@echo "Fetching atomic data from OpenADAS"
	@echo ""
	python fetch_adas_data.py &> fetch_adas_data_log.txt
	@echo ""
	@echo "Atomic data successfully retrieved"
	@echo "see fetch_adas_data_log.txt for build output and warnings/errors"
	@echo ""

setup: setup_fortran_programs.py
	@echo "Building fortran helper functions in src"
	@echo ""
	python setup_fortran_programs.py build_ext --inplace &> setup_fortran_programs_log.txt
	@echo ""
	@echo "Fortran functions successfully built"
	@echo "see setup_fortran_programs_log.txt for build output and warnings/errors"
	@echo ""

clean:
	@echo "Cleaning atomic1D directory to fresh-install state"
	@echo "-> excluding adas_data and src tar files retrieved in fetch"
	rm -rf __pycache__
	rm -rf build
	rm -rf atomic/_xxdata_11.cpython*
	rm -rf atomic/_xxdata_15.cpython*
	rm -rf src/_xxdata_11module.c
	rm -rf src/_xxdata_15module.c
	rm -f setup_fortran_programs_log.txt
	@echo ""
	@echo "atomic1D directory cleaned (except fetch)"
	@echo ""

clean_refetch:
	@echo "Cleaning atomic1D directory to fresh-install state"
	@echo "-> including deleting adas_data and src functions"
	@echo ""
	rm -rf adas_data
	rm -rf __pycache__
	rm -rf build
	rm -rf atomic/_xxdata_11.cpython*
	rm -rf atomic/_xxdata_15.cpython*
	rm -rf src/xxdata_11
	rm -rf src/xxdata_15
	rm -rf src/_xxdata_11module.c
	rm -rf src/_xxdata_15module.c
	rm -f src/xxdata_11.tar.gz
	rm -f src/xxdata_15.tar.gz
	rm -f fetch_adas_data_log.txt
	rm -f setup_fortran_programs_log.txt
	@echo ""
	@echo "atomic1D directory cleaned"
	@echo ""