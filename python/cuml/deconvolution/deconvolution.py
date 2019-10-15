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
from scipy.signal import convolve2d, fftconvolve

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
    for i in range(maxiter):
        utp1 = ut * convolve_method(d / convolve_method(ut,psf, 'same'), psf[::-1, ::-1], 'same')

        diff_norm_u = np.linalg.norm(utp1-ut)

        if disp > 0:
            if disp > 100:
                print("{}:|ut-u_t+1|=".format(i), diff_norm_u)
            if np.mod(i, disp) == 0:
                print("{}:|ut-u_t+1|=".format(i), diff_norm_u)
            
        if disp > 0:
            if i == maxiter:
                print("STOPPING DUE TO MAXIMUM NUMBER OF ITERATIONS: ", maxiter)

        ut = utp1

    return utp1


def deconvolution_fmin(d, psf,
                       maxiter=10,
                       stop_delta_u=1e-1,
                       maxiter=10,
                       stop_grad=1e-4,
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

    sigma = 0.5


    def f(I):
        return np.linalg.norm(d - convolve_method(I, psf, 'same'))**2 / (2*sigma)
        
    def g(I):
        diff = d - convolve_method(I, psf, 'same')
        return convolve_method(diff, psf[::-1,::-1])/sigma



