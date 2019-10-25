from libcpp.vector cimport vector
from libcpp.functional cimport function
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.string cimport string
cimport cython

import numpy as np
cimport numpy as np

from cpython cimport PyObject, Py_INCREF

cdef extern from "optimization/lbfgs_b.h" namespace "MLCommon::Optimization":
  ctypedef void (*doublefun)(double* x, void* f)
  void testf(doublefun wrapper, void* f);


def f(x):
    print("x=", x)

cdef void fun(double* x, void* f):
    
    cdef np.npy_intp shape[2]
    cdef np.npy_intp size = 1
    cdef np.ndarray[double, ndim=1] x2 = np.zeros(10)
    shape[0] = size
    _ndarray = np.PyArray_SimpleNewFromData(1, &size, np.NPY_DOUBLE, &x2[0])
    nparray = np.array(_ndarray, copy=False)
    (<object>f)(nparray)
    Py_INCREF(nparray)


def run():

    testf(fun, <void*>f)
