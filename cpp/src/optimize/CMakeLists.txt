project(OptimizeLib CXX Fortran)

find_package(BLAS REQUIRED)

add_library(lbfgsb_f
  # minpack2/dcsrch.f
  # minpack2/dcstep.f
  lbfgsb_src/lbfgsb.f
  lbfgsb_src/linpack.f
  lbfgsb_src/timer.f
  )

add_library(lbfgsb
  lbfgs_b.cpp
  lbfgsb_f_interface.cpp
  linesearch.cpp
  f_util.cpp
  )

target_link_libraries(lbfgsb_f
  gfortran
  ${BLAS_LIBRARIES}
  )

target_link_libraries(lbfgsb
  lbfgsb_f
  )

target_include_directories(lbfgsb
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
