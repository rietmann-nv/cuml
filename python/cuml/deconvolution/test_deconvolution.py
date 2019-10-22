import numpy as np

import matplotlib.pyplot as plt
# from skimage import color, data, restoration
from scipy.signal import convolve2d

# from cuml.deconvolution.deconvolution import richardson_lucy
from deconvolution import richardson_lucy, richardson_lucy_gpu
from deconvolution import deconvolution_fmin, deconvolution_fmin_poisson
from deconvolution import cupy_fftconvolve

from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve

from timeit import default_timer as timer

import cupy as cp

from PIL import Image
import scipy

from IPython.core.debugger import set_trace

def loadpng(filename, as_gray=False):
    """
    Loads a PNG with option to convert to grayscale
    """
    img = Image.open(filename)
    if as_gray:
        return np.array(img.convert('L'))

    return np.array(img)/256.0


def astronaut():
    """Load astronaut image in grayscale"""
    return loadpng("images/astronaut.png", as_gray=True)


def test_fftconvolve_gpu(plot=False):
    # astro = color.rgb2gray(astronaut())
    astro = astronaut()
    # astro = astro[::12,::12]

    psf = np.ones((5, 5)) / 25
    d = fftconvolve(astro.copy(), psf.copy(), 'same')

    d2_gpu = cupy_fftconvolve(astro.copy(), psf.copy())
    d2_cpu = cp.asnumpy(d2_gpu)

    np.testing.assert_allclose(d, d2_cpu)

    if plot:
        f, ax = plt.subplots(1, 2, figsize=(8, 5), num=1, clear=True)

        plt.gray()
        ax[0].imshow(d)
        ax[0].set_title("FFTCONVOLVE\nCPU")
        ax[1].imshow(d2_cpu)
        ax[1].set_title("FFTCONVOLVE\nGPU")
        f.subplots_adjust(wspace=0.02, hspace=0.2,
                          top=0.9, bottom=0.05, left=0, right=1)



def test_fmin_poisson():

    # astro = color.rgb2gray(data.astronaut())
    astro = astronaut()
    # astro = astro[::25,::25]

    psf = np.ones((5, 5)) / 25
    d = convolve2d(astro, psf, 'same')

    # poisson noise assumption is common, particularly in astronomy (CCD sensors)
    d += (np.random.poisson(25, size=d.shape) - 10) / 255
    d = np.abs(d)
    # set_trace()
    astro_deconv = deconvolution_fmin_poisson(d.copy(), psf.copy(), maxiter=20, disp=1)
    # return
    f, ax = plt.subplots(1, 3, figsize=(8, 5), num=1, clear=True)
    plt.gray()
    ax[0].imshow(astro)
    ax[0].set_title("Orig.")
    ax[1].imshow(d, vmin=astro.min(), vmax=astro.max())
    ax[1].set_title("Conv(psf) + noise")
    ax[2].imshow(astro_deconv, vmin=astro.min(), vmax=astro.max())
    ax[2].set_title("My R-L Deconv")

    f.subplots_adjust(wspace=0.02, hspace=0.2,
                      top=0.9, bottom=0.05, left=0, right=1)

def test_grad():

    astro = astronaut()
    astro = astro[::50, ::50]
    psf = np.ones((5, 5)) / 25
    d = convolve2d(astro, psf, 'same')
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    lambda_reg=0.01

    def f(Of):
        reg_term = lambda_reg/2.0 * np.linalg.norm(convolve2d(d, kernel, 'same'))**2
        O = np.reshape(Of, d.shape)
        ll_all = d * np.log(convolve2d(O, psf, 'same')) - convolve2d(O, psf, 'same')
        return -ll_all.sum()

    def g(Of):
        O = np.reshape(Of, d.shape)
        gll = convolve2d(d/convolve2d(O, psf, 'same'), psf, 'same') - convolve2d(np.full(d.shape, 1), psf, 'same')
        gll2 = d/convolve2d(O, psf, 'same') * convolve2d(np.full(d.shape, 1), psf, 'same') - convolve2d(np.full(d.shape, 1), psf, 'same')
        # reg_term = lambda_reg * convolve_method(convolve_method(I, kernel, 'same'), kernel, 'same')
        return -gll.ravel()

    gk_fd = scipy.optimize.optimize._approx_fprime_helper(d.ravel(), f, 1e-5)
    gk = g(d.ravel())

    # print("f(d)=", f(d.ravel()))
    print("gk_fd=", gk_fd)
    print("gk_an=", gk)
    # print("gkfd-gkan=", gk_fd - gk)
    # print("[1:5](gk, gk_fd)=\n",(gk_fd[0:5], gk[0:5]))
    np.testing.assert_allclose(gk, gk_fd)
                                                         

def test_deconvolve():
    astro = astronaut()
    # astro = color.rgb2gray(data.astronaut())
    # astro = astro[::25,::25]

    psf = np.ones((5, 5)) / 25
    d = convolve2d(astro, psf, 'same')

    # poisson noise assumption is common, particularly in astronomy (CCD sensors)
    d += (np.random.poisson(25, size=d.shape) - 10) / 255

    d_gpu = cp.array(d)
    d_psf = cp.array(psf)
    start = timer()
    astro_deconv_gpu = richardson_lucy_gpu(d_gpu, d_psf)
    end = timer()
    h_astro_deconv_gpu = cp.asnumpy(astro_deconv_gpu)
    time_gpurl_s = end - start
    num_iter = 10
    start = timer()
    astro_deconv = richardson_lucy(d.copy(), psf.copy(), disp=-1, maxiter=num_iter)
    end = timer()
    time_myrl_s = end - start

    np.testing.assert_allclose(astro_deconv, h_astro_deconv_gpu)
    

    astro_deconv3 = deconvolution_fmin(d.copy(), psf.copy(), maxiter=num_iter, disp=1)
    astro_deconv4 = deconvolution_fmin_poisson(d.copy(), psf.copy(), maxiter=num_iter, disp=1)

    # start = timer()
    # astro_deconv2 = restoration.richardson_lucy(d.copy(), psf.copy(), iterations=num_iter)
    # end = timer()
    # time_sprl_s = end - start

    # for 512x512 on CPU
    # Time My R-L: 0.29476031521335244s, Scikit-im R-L: 0.28734008595347404s
    # GPU R-L: 0.05041124401031993s
    print("Time My R-L: {}s, GPU R-L: {}s".format(time_myrl_s, time_gpurl_s))

    # astro_deconv3_conv = convolve2d(astro_deconv3, psf, 'same')

    f, ax = plt.subplots(1, 6, figsize=(8, 5), num=1, clear=True)

    for a in (ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]):
       a.axis('off')

    plt.gray()
    ax[0].imshow(astro)
    ax[0].set_title("Orig.")
    ax[1].imshow(d, vmin=astro.min(), vmax=astro.max())
    ax[1].set_title("Conv(psf) + noise")
    # ax[1].imshow(astro_deconv3_conv, vmin=astro.min(), vmax=astro.max())
    # ax[1].set_title("Fmin Deconv\n + Conv(psf)")
    ax[2].imshow(astro_deconv, vmin=astro.min(), vmax=astro.max())
    ax[2].set_title("My R-L Deconv")
    ax[3].imshow(h_astro_deconv_gpu, vmin=astro.min(), vmax=astro.max())
    ax[3].set_title("GPU R-L Deconv")

    # ax[3].imshow(astro_deconv2, vmin=astro.min(), vmax=astro.max())
    # ax[3].set_title("Sk-Image\nDeconv")
    ax[4].imshow(astro_deconv3, vmin=astro.min(), vmax=astro.max())
    ax[4].set_title("Fmin Gauss\n Deconv")

    ax[5].imshow(astro_deconv4, vmin=astro.min(), vmax=astro.max())
    ax[5].set_title("Fmin Poiss\n Deconv")

    f.subplots_adjust(wspace=0.02, hspace=0.2,
                      top=0.9, bottom=0.05, left=0, right=1)
    f.show()
    
def orig():
    """ Taken from sk-image's documentation on deconvolution"""
    astro = color.rgb2gray(data.astronaut())

    psf = np.ones((5, 5)) / 25
    astro = conv2(astro, psf, 'same')
    # Add Noise to Image
    astro_noisy = astro.copy()
    astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

    # Restore Image using Richardson-Lucy algorithm
    deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)
    deconvolved_RL2 = richardson_lucy(astro_noisy.copy(), psf.copy(), disp=1, maxiter=30)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2], ax[3]):
           a.axis('off')

    ax[0].imshow(astro)
    ax[0].set_title('Original Data')

    ax[1].imshow(astro_noisy)
    ax[1].set_title('Noisy data')

    ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[2].set_title('Restoration using\nRichardson-Lucy')

    ax[3].imshow(deconvolved_RL2, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[3].set_title('Restoration using\nRichardson-Lucy2')


    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
