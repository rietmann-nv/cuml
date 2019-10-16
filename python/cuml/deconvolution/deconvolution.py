#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import cupy as cp
from scipy.signal import convolve2d, fftconvolve
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize as optimize
import scipy
from IPython.core.debugger import set_trace
from scipy import fftpack


def cupy_fftconvolve(h_I, h_psf, mode="same"):
    """Limited CUPY port of scipy.signal.fftconvolve"""
    if mode is not "same":
        raise Exception("Only 'same' mode supported for now")

    I = cp.array(h_I)
    psf = cp.array(h_psf)

    s1 = np.array(I.shape)
    s2 = np.array(psf.shape)
    shape = np.maximum(s1, s2)
    shape = s1 + s2 - 1
    fshape = (shape[0], shape[1])

    axes = np.array([0, 1])

    sp1 = cp.fft.rfftn(I, fshape, axes=axes)
    sp2 = cp.fft.rfftn(psf, fshape, axes=axes)
    ret = cp.fft.irfftn(sp1 * sp2, fshape, axes=axes)

    if mode is "same":
        ret = scipy.signal.signaltools._centered(ret, I.shape)

    return ret
    
def richardson_lucy(d,
                    psf,
                    stop_delta_u=1e-8,
                    maxiter=10,
                    disp=-1):
    """
    Computes the deconvolution of the data `d` created with `psf` using the
    Richardson-Lucy algorithm.

    Arguments
    ---------
    d            : array
                   recorded data (d = I + e)
    psf          : array
                   point spread kernel
    stop_delta_u : float (default 1e-8)
                   stop when ||u_{t+1} - u_t|| < 'stop_delta_u'
    maxiter      : int (default 10)
                   maximumum number of R-L iterations
    disp         : int (default=-1)
                   verbosity level;
                   -1    = quiet
                   1-100 = display info every `disp` iterations
                   >100  = maximum verbosity
    Returns
    --------
    image : array
            deconvolved image

    Algorithm
    ---------
    u (*) psf = d = I + eps
    where
    u: final image ("output")
    psf: point spread function
    d: recorded data ("input")
    I: recorded image minus noise
    eps: noise

    Richardson-Lucy algorithm:
    u_{t+1} = u_t (d / (u_t (*) psf) (*) psf^* )

    where ^* is a vertical + horizontal mirror

    """
    # Note: convolve optimization taken from skimage's richardson_lucy implementation
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(d.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in d.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(d.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    # ut is a constant 0.5 to be consistent with skimage.
    # a more logical choice would simply be ut = d.copy()
    ut = np.full(d.shape, 0.5)
    utp1 = ut
    psf_flip = psf[::-1, ::-1]
    for i in range(maxiter):
        utp1 = ut * convolve_method(d / convolve_method(ut,psf, 'same'), psf_flip, 'same')
        ut = utp1

        if disp > 0:
            diff_norm_u = np.linalg.norm(utp1-ut)
            if disp > 100:
                print("{}:|ut-u_t+1|=".format(i), diff_norm_u)
            if np.mod(i, disp) == 0:
                print("{}:|ut-u_t+1|=".format(i), diff_norm_u)

        if disp > 0:
            if i == maxiter:
                print("STOPPING DUE TO MAXIMUM NUMBER OF ITERATIONS: ", maxiter)

    return utp1


def deconvolution_fmin(d, psf,
                       maxiter=10,
                       stop_delta_u=1e-1,
                       stop_grad=1e-4,
                       lambda_reg=0.01,
                       disp=-1):
    """Computes the deconvolution of the data `d` created with `psf` using an
    iterative numerical optimization technique.

    Arguments
    ---------
    d            : array
                   recorded data (d = I + e)
    psf          : array
                   point spread kernel
    maxiter      : int (default 10)
                   maximumum number of R-L iterations
    stop_delta_u : float (default 1e-8)
                   stop when ||u_{t+1} - u_t|| < 'stop_delta_u'
    stop_grad    : float (default 1e-4)
                   Minimization stopping criterion when gradient "flat enough"
    lambda_reg   : float (default 0.01)
                   Regularization parameter weight that damps high-frequency noise.
    disp         : int (default=-1)
                   verbosity level;
                   -1    = quiet
                   1-100 = display info every `disp` iterations
                   >100  = maximum verbosity
    Returns
    --------
    image : array
            deconvolved image

    Algorithm
    ---------

    Instead of an explicitly iterative algorithm such as Richardson-Lucy, this
    method frames the deconvolution as a numerical minimization (which is
    implicitly iterative).

    Consider the functional ("loss") J(I)

              ||d - P (x) I||^2     l
       J(I) = ---------------    + --- ||H (x) I||^2
                  2 sigma           2

    and its gradient G(J(I))

       G(J(I)) = P^* (x) (D - P (x) I) + l H (x) (H (x) I)

    where `l` is the regularization penalty and `H` is a high-pass filter matrix.

    By passing this into an L-BFGS minimization routine, we retrieve a
    deconvolved image. Raising or lowering `l` allows the user to increase or
    decrease noise reduction (respectively).

    """

    # Note: convolve optimization taken from skimage's richardson_lucy implementation
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(d.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in d.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(d.shape) > 2:
        if disp > 0:
            print("Using FFTCONVOLVE")
        convolve_method = fftconvolve
    else:
        if disp > 0:
            print("Using convolve2d")
        convolve_method = convolve2d

    sigma = 0.8
    # regularization high-pass from:
    # https://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    def f(If):
        I = np.reshape(If, d.shape)
        reg_term = lambda_reg/2.0 * np.linalg.norm(convolve_method(I, kernel))**2
        return np.linalg.norm(d - convolve_method(I, psf, 'same'))**2 / (2*sigma) + reg_term

    def g(If):
        I = np.reshape(If, d.shape)
        diff = d - convolve_method(I, psf, 'same')
        reg_term = lambda_reg * convolve_method(convolve_method(I, kernel, 'same'), kernel, 'same')
        return -(convolve_method(diff, psf[::-1,::-1], 'same')/sigma).ravel() + reg_term.ravel()

    x0 = d.ravel()
    image, f, res_info = fmin_l_bfgs_b(f, x0, fprime=g, maxiter=maxiter, disp=disp, pgtol=stop_grad)
    return image.reshape(d.shape)
