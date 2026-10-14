if(PythonSupport)
  # Standalone MPI benchmark environment; no test-suite fixture imports.
  configure_file(mpi_environment.json.in mpi_environment.json @ONLY)
endif()
