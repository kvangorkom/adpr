from functools import partial
from itertools import product

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from .utils import get_scaled_coords

class ForwardModel(object):
    def __init__(self, Np, Nf, fnum, pix_m, reference_wavelength, wavelengths, dz):
        '''
        To do: generalize this to specify model in different ways
        (pass in q rather than pix_m, RMS defocus or waves rather than dz, etc)
        '''
        self.Np = Np
        self.Nf = Nf
        self.fnum = fnum
        self.pix_m = pix_m
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength
        self.dz = dz
        self.q = get_q(reference_wavelength, fnum, pix_m)

        self.Mx = self.My = self.fresnel_TFs = self.pupil_coord = None
        
        self.precompute_quantities()
        
    def precompute_quantities(self):
        '''
        Precompute quantities needed for the forward propagation
        '''
        
        # pupil coordinates are independent of wavelength
        pupil_coord = get_pupil_coords(self.Np) # wavelength-independent

        fresnel_TFs_wave = []
        Mx_wave = []
        My_wave = []
        for wave in self.wavelengths:
            # get wavelength-dependent focal coordinates
            fc_wave = get_focal_coords(self.Nf, wave, self.fnum, self.pix_m)

            # MFT
            Mx, My = get_MFT_matrices(fc_wave, pupil_coord)
            Mx_wave.append(Mx / (1j*wave)) # double-check this scaling
            My_wave.append(My / (1j*wave))

            # get wavelength-dependent fresnel TFs
            fresnel_TFs_wave.append( jnp.array([get_fresnel_TF(z, self.Np, wave, self.fnum) for z in self.dz]) )

        # is there a good reason to throw these in a dict rather than be object properties?
        self.Mx = Mx_wave
        self.My = My_wave
        self.fresnel_TFs = fresnel_TFs_wave
        self.pupil_coord = pupil_coord

@jax.jit
def forward_propagate(pupil, opd, wavelengths, fresnel_TFs, Mx, My):
    allI = []
    for i, wave in enumerate(wavelengths):
        phase = 2 * jnp.pi / wave * opd
        allI.append(jnp.abs(propagate_MFT( fresnel_TFs[i] * pupil * jnp.exp(1j*phase),
                                Mx[i], My[i]))**2)
    return jnp.sum(jnp.array(allI),axis=0)

def propagate_MFT(E, Mx, My):
    ''' This is the dz x X x Y version'''
    return jnp.einsum('ijk,lj->ikl', jnp.einsum('ijk,kl->ijl', E, Mx), My)

def get_q(wavelength, fnum, pix_m):
    return wavelength * fnum / pix_m

def get_focal_coords(N, wavelength, fnum, pix_m):
    '''
    Get detector pixel coordinates in terms of lambda/D oversampling factor q
    '''
    q = get_q(wavelength, fnum, pix_m)
    return (jnp.linspace(-(N-1)/2.0, (N-1)/2.0, num=N) - 0.5) / q

def get_pupil_coords(N):
    '''
    Get normalized pupil coordinates (D=1)
    '''
    return (jnp.linspace(-(N-1)/2.0, (N-1)/2.0, num=N) - 0.5) / N

def get_fresnel_TF(dz, N, wavelength, fnum):
    '''
    Get the Fresnel transfer function for a shift dz from focus
    '''
    df = 1.0 / (N * wavelength * fnum)
    rp = get_scaled_coords(N,df, shift=False)[-1]
    return jnp.exp(-1j*jnp.pi*dz*wavelength*(rp**2))

def get_MFT_matrices(focal_coord, pupil_coord):
    vy = jnp.outer(focal_coord, pupil_coord)
    xu = jnp.outer(pupil_coord, focal_coord)
    
    # check me
    norm = jnp.sqrt(len(focal_coord)**2 + len(pupil_coord)**2)
    My = jnp.exp(-1j*2*jnp.pi*vy) / norm 
    Mx = jnp.exp(-1j*2*jnp.pi*xu) / norm
    
    return Mx, My