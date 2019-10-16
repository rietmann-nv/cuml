import numpy as np

import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d

# from cuml.deconvolution.deconvolution import richardson_lucy
from deconvolution import richardson_lucy, deconvolution_fmin, cupy_fftconvolve

from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve

from timeit import default_timer as timer

import cupy as cp

def test_fftconvolve_gpu(plot=False):
    astro = color.rgb2gray(data.astronaut())
    astro = astro[::25,::25]

    psf = np.ones((5, 5)) / 25
    d = fftconvolve(astro.copy(), psf.copy(), 'same')

    d2_gpu = cupy_convolve(astro.copy(), psf.copy())
    d2_cpu = cp.asnumpy(d2_gpu)

    np.testing.assert_allclose(d, d2_cpu)

    if plot:
        f, ax = plt.subplots(1, 2, figsize=(8, 5), num=1, clear=True)

        plt.gray()
        ax[0].imshow(d)
        ax[0].set_title("FFTCONVOLVE\nCPU")
        ax[1].imshow(d_gpu)
        ax[1].set_title("FFTCONVOLVE\nGPU")
        f.subplots_adjust(wspace=0.02, hspace=0.2,
                          top=0.9, bottom=0.05, left=0, right=1)


def test_sklearn():
    astro = color.rgb2gray(data.astronaut())
    # astro = astro[::25,::25]

    psf = np.ones((5, 5)) / 25
    d = convolve2d(astro, psf, 'same')

    # poisson noise assumption is common, particularly in astronomy (CCD sensors)
    d += (np.random.poisson(25, size=d.shape) - 10) / 255

    num_iter = 10
    astro_deconv3 = deconvolution_fmin(d.copy(), psf.copy(), maxiter=2*num_iter, disp=1)
    start = timer()
    astro_deconv = richardson_lucy(d.copy(), psf.copy(), disp=-1, maxiter=num_iter)
    end = timer()
    time_myrl_s = end - start
    start = timer()
    astro_deconv2 = restoration.richardson_lucy(d.copy(), psf.copy(), iterations=num_iter)
    end = timer()
    time_sprl_s = end - start

    # for 512x512 on CPU
    # Time My R-L: 0.29476031521335244s, Scikit-im R-L: 0.28734008595347404s
    print("Time My R-L: {}s, Scikit-im R-L: {}s".format(time_myrl_s, time_sprl_s))

    astro_deconv3_conv = convolve2d(astro_deconv3, psf, 'same')

    f, ax = plt.subplots(1, 5, figsize=(8, 5), num=1, clear=True)

    for a in (ax[0], ax[1], ax[2], ax[3], ax[4]):
       a.axis('off')

    plt.gray()
    ax[0].imshow(astro)
    ax[0].set_title("Orig.")
    ax[1].imshow(d, vmin=astro.min(), vmax=astro.max())
    ax[1].set_title("Conv(psf) + noise")
    ax[2].imshow(astro_deconv3_conv, vmin=astro.min(), vmax=astro.max())
    ax[2].set_title("Fmin Deconv\n + Conv(psf)")
    ax[3].imshow(astro_deconv, vmin=astro.min(), vmax=astro.max())
    ax[3].set_title("My R-L Deconv")
    # ax[3].imshow(astro_deconv2, vmin=astro.min(), vmax=astro.max())
    # ax[3].set_title("Sk-Image\nDeconv")
    ax[4].imshow(astro_deconv3, vmin=astro.min(), vmax=astro.max())
    ax[4].set_title("Fmin Deconv")

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
