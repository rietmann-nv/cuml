project(OptimizeLib CXX Fortran)

find_package(BLAS REQUIRED)

add_library(lbfgsb
  lbfgs_b.cpp
  lbfgsb_f_interface.cpp
  minpack2/dcsrch.f
  minpack2/dcstep.f
  lbfgsb_src/lbfgsb.f
  lbfgsb_src/linpack.f
  lbfgsb_src/timer.f
  )

target_link_libraries(lbfgsb
  gfortran
  ${BLAS_LIBRARIES}
  )
