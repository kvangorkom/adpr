import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import numpy as np
from skimage.transform import downscale_local_mean

def fft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)
        
    if shift:
        shiftfunc = jnp.fft.fftshift
        ishiftfunc = jnp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    t = jnp.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
    return shiftfunc(t,axes=axes)

def ifft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)

    if shift:
        shiftfunc = jnp.fft.fftshift
        ishiftfunc = jnp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    t = jnp.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
    return shiftfunc(t, axes=axes)

def gauss_convolve(arr, gauss):
    return ifft2_shiftnorm(fft2_shiftnorm(arr) * gauss).real

def get_inscribed_circular_pupil(N, upsample=2):
    '''
    Pupil image stays the same, pupil coordinates scale with wavelength
    '''
    rcoords = np.array(get_scaled_coords(N*upsample, 1, shift=False)[-1])
    return jnp.array(downscale_local_mean(rcoords < (N-1)*upsample/2, upsample))

def dz_to_RMS_defocus(dz, fnum):
    # axial displacement dz (in meters) to RMS defocus in the pupil (also in meters)
    return dz / ( jnp.sqrt(3) * 16 * (fnum**2) )

def RMS_defocus_to_dz(rms, fnum):
    # RMS defocus in the pupil (meters) to axial displacement dz (in meters)
    return rms * jnp.sqrt(3) * 16 * (fnum**2) 

def get_scaled_coords(N, scale, center=True, shift=True):
    if center:
        cen = (N-1)/2.0
    else:
        cen = 0
        
    if shift:
        shiftfunc = jnp.fft.fftshift
    else:
        shiftfunc = lambda x: x
    cy, cx = (shiftfunc(jnp.indices((N,N))) - cen) * scale
    r = jnp.sqrt(cy**2 + cx**2)
    return [cy, cx, r]

def get_gauss(sigma, shape, cenyx=None):
    if cenyx is None:
        cenyx = jnp.asarray([(shape[0])/2., (shape[1])/2.]) # no -1
    yy, xx = jnp.indices(shape).astype(float) - cenyx[:,None,None]
    g = jnp.exp(-0.5*(yy**2+xx**2)/sigma**2)
    return g / jnp.sum(g)