# Configure the version
configure_file(version.py.in version.py)
configure_file(_backend_config.py.in _backend_config.py @ONLY)

# Install the version file and the pure Python dependencies
set(PYTHON_LIB_DEST ${TRIQS_PYTHON_LIB_DEST_ROOT}/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py ${CMAKE_CURRENT_BINARY_DIR}/_backend_config.py DESTINATION ${PYTHON_LIB_DEST})
install(DIRECTORY ${adapol_SOURCE_DIR}/src/adapol DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT} FILES_MATCHING PATTERN "*.py")
install(DIRECTORY ${pyed_SOURCE_DIR}/pyed DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT} FILES_MATCHING PATTERN "*.py")

xca_add_python_package(triqs_xca LIBRARIES triqs_xca_c)
