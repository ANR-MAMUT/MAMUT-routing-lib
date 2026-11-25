# Header-only exact TD route evaluator, linked to the consumer's GMP installation.
if(TARGET MAMUTRouting::td_exact)
    return()
endif()

find_path(MAMUTRouting_GMPXX_INCLUDE_DIR gmpxx.h)
find_path(MAMUTRouting_GMP_INCLUDE_DIR gmp.h)
find_library(MAMUTRouting_GMPXX_LIBRARY NAMES gmpxx)
find_library(MAMUTRouting_GMP_LIBRARY NAMES gmp)
if(NOT MAMUTRouting_GMPXX_INCLUDE_DIR OR NOT MAMUTRouting_GMP_INCLUDE_DIR
   OR NOT MAMUTRouting_GMPXX_LIBRARY OR NOT MAMUTRouting_GMP_LIBRARY)
    set(MAMUTRouting_FOUND FALSE)
    set(MAMUTRouting_NOT_FOUND_MESSAGE "The exact TD kernel needs GMP and its C++ headers and libraries.")
    return()
endif()

add_library(MAMUTRouting::td_exact INTERFACE IMPORTED)
set_target_properties(MAMUTRouting::td_exact PROPERTIES
    INTERFACE_COMPILE_FEATURES cxx_std_17
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../include;${MAMUTRouting_GMPXX_INCLUDE_DIR};${MAMUTRouting_GMP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${MAMUTRouting_GMPXX_LIBRARY};${MAMUTRouting_GMP_LIBRARY}"
)
