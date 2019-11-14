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
import matplotlib.pyplot as plt

import jax.numpy as jp
import jax

def cupy_fftconvolve(u_I, u_psf, mode="same"):
    """Limited CUPY port of scipy.signal.fftconvolve"""
    if mode is not "same":
        raise Exception("Only 'same' mode supported for now")

    I = cp.array(u_I)
    psf = cp.array(u_psf)

    s1 = np.array(I.shape)
    s2 = np.array(psf.shape)
    shape = np.maximum(s1, s2)
    shape = s1 + s2 - 1
    fshape = (shape[0], shape[1])

    axes = np.array([0, 1])
    
    sp1 = cp.fft.rfftn(I, fshape, axes=axes)
    sp2 = cp.fft.rfftn(psf, fshape, axes=axes)
    ret = cp.fft.irfftn(sp1 * sp2, fshape, axes=axes)

    # if mode is "same":
        # ret = scipy.signal.signaltools._centered(ret, I.shape)

    return ret

def jax_fftconvolve(u, psf):

    s1 = np.array(I.shape)
    s2 = np.array(psf.shape)
    shape = np.maximum(s1, s2)
    shape = s1 + s2 - 1
    fshape = (shape[0], shape[1])

    sp1 = jp.fft.fftn(u, fshape, axes=)

def jax_convolve(u, u2, psf):
    (h, w) = u.shape
    (hp, wp) = psf.shape
    assert hp == wp

    # NCHW format
    u_jax = np.zeros((2,1,h,w))
    u_jax[0,0,:,:] = u[:,:]
    u_jax[1,0,:,:] = u2[:,:]
    # IOHW format
    psf_jax = np.zeros((1, 1, hp, wp))
    psf_jax[0, 0, :, :] = psf[:, :]
    uc_jax = jax.lax.conv(u_jax, psf_jax, (1,1), 'SAME')

    # put back into HW format
    uc = np.zeros((h, w))
    uc[:, :] = uc_jax[0, 0, :, :]
    uc2 = np.zeros((h, w))
    uc2[:, :] = uc_jax[1, 0, :, :]

    return uc, uc2

# def richardson_lucy_jax(d, psf, maxiter=10):
#     (h, w) = d.shape
#     ut = jp.full((1, 1, h, w), 0.5)
#     utp1 = ut
#     d_jp = jp.zeros((1, 1, h, w))
#     d_jp[0,0,:,:] = d[:,:]
#     psf_jp = jp.zeros()
#     for _ in range(maxiter):
#         d_ut_psf = d_jp / jax.lax.conv(ut, psf_jp, (1,1), 'SAME')
#         utp1 = ut * jax.lax.conv(d_ut_psf, psf_jp, (1,1), 'SAME')


def richardson_lucy_gpu(d,
                        psf,
                        stop_delta_u=1e-8,
                        maxiter=10,
                        disp=-1):

    ut = cp.full(d.shape, 0.5)
    utp1 = ut

    for _ in range(maxiter):
        utp1 = ut * cupy_fftconvolve(d / cupy_fftconvolve(ut, psf, 'same'), psf, 'same')
        ut = utp1

    return utp1


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
    iterative numerical optimization technique that maximizes a likelihood
    assuming a gaussian noise distribution.

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
        reg_term = lambda_reg/2.0 * np.linalg.norm(convolve_method(I, kernel, 'same'))**2
        return np.linalg.norm(d - convolve_method(I, psf, 'same'))**2 / (2*sigma) + reg_term

    def g(If):
        I = np.reshape(If, d.shape)
        diff = d - convolve_method(I, psf, 'same')
        reg_term = lambda_reg * convolve_method(convolve_method(I, kernel, 'same'), kernel, 'same')
        return -(convolve_method(diff, psf[::-1,::-1], 'same')/sigma).ravel() + reg_term.ravel()

    x0 = d.ravel()
    image, f, res_info = fmin_l_bfgs_b(f, x0, fprime=g, maxiter=maxiter, disp=disp, pgtol=stop_grad)
    return image.reshape(d.shape)


def deconvolution_fmin_poisson(d, psf,
                               maxiter=10,
                               stop_delta_u=1e-1,
                               stop_grad=1e-4,
                               lambda_reg=0.00001,
                               disp=-1):
    """Computes the deconvolution of the data `d` created with `psf` using an
    iterative numerical optimization technique that assumes a likelihood with a
    poissonian noise distribution.

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
    This uses a likelihood that assumes a poissonian distribution.

    """

    convolve_method = fftconvolve

    # regularization high-pass from:
    # https://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
    kernel = np.array([[-1., -1., -1.],
                       [-1.,  8., -1.],
                       [-1., -1., -1.]])

    def f(Of):
        O = np.reshape(Of, d.shape)
        reg_term = lambda_reg/2.0 * np.linalg.norm(convolve_method(O, kernel, 'same'))**2
        O_star_psf = convolve_method(O, psf, 'same')
        O_star_psf = np.abs(O_star_psf)
        ll_all = d * np.log(O_star_psf) - O_star_psf
        return -ll_all.sum() + reg_term

    def g(Of):
        O = np.reshape(Of, d.shape)
        O_star_psf = convolve_method(O, psf, 'same')
        gll = convolve_method(d/O_star_psf, psf, 'same') - convolve_method(np.full(d.shape, 1.0), psf, 'same')
        reg_term = lambda_reg * convolve_method(convolve_method(O, kernel, 'same'), kernel, 'same')
        # return -gll.ravel() + reg_term.ravel()
        gll_an = -gll.ravel() + reg_term.ravel()
        return gll_an
        # return -gll.ravel()
    # gll_fd = scipy.optimize.optimize._approx_fprime_helper(Of, f, 1e-5)
        # print("|gll_an - gll_fd|=", np.linalg.norm(gll_an - gll_fd))
        # print("{} vs {}", gll_an, gll_fd)
        # return gll_fd
        

    x0 = d.ravel()
    # gll_fd = scipy.optimize.optimize._approx_fprime_helper(x0, f, 1e-5)
    # gll_fd_img = np.reshape(gll_fd, d.shape)
    # gll_an = g(x0)
    # gll_an_img = np.reshape(gll_an, d.shape)
    # f, ax = plt.subplots(1, 3, figsize=(8, 5), num=1, clear=True)
    # plt.gray()
    # ax[0].imshow(gll_fd_img)
    # ax[0].set_title("FD.")
    # ax[1].imshow(gll_an_img)
    # ax[1].set_title("Analytical")
    # ax[2].imshow(d)
    # ax[2].set_title("Orig")
    # return
    bounds = [(0, None)]*x0.size
    image, f, res_info = fmin_l_bfgs_b(f, x0, fprime=g,
                                       bounds=bounds,
                                       maxiter=maxiter, disp=disp, pgtol=stop_grad)
    return image.reshape(d.shape)
