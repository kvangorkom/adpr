from functools import partial

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import value_and_grad

from jaxopt import LBFGS, LBFGSB

from .utils import get_gauss, gauss_convolve
from .model import forward_propagate

LBFGSB_kwargs = {'maxls' : 100, 'history_size' : 10, 'tol' : 1e-4, 'min_stepsize' : 0.1e-9, 'implicit_diff' : False}

class Estimation(object):
    
    def __init__(self, forward_model, estimate_phase=True, estimate_amplitude=False, estimate_spectrum=False, estimate_bg=False, phase_modal=False, amplitude_modal=False, wreg=1e-2, modes=None, modes_tiptilt=None, method=LBFGSB, method_kwargs=LBFGSB_kwargs, maxiter=100):
        self.forward_model = forward_model
        self.method = method
        self.method_kwargs = method_kwargs
        self.wreg = wreg
        self.model = forward_model
        
        self.estimate_phase = estimate_phase
        self.estimate_amplitude = estimate_amplitude
        self.estimate_spectrum = estimate_spectrum
        self.estimate_bg = estimate_bg
        self.phase_modal = phase_modal
        self.amplitude_modal = amplitude_modal
        
        self.modes = modes
        if modes is not None:
            self.nmodes = len(modes)
        else:
            self.nmodes = None

        self.modes_tiptilt = modes_tiptilt
        
        # to do: don't hardcode these things!
        self.gauss_init = 0.1
        self.gauss_final = 20.0
        
        self.maxiter = maxiter
        
        self.get_init()
        
    def get_init(self):
        init = []
        bounds = [[],[]]
        if self.estimate_phase:
            if self.phase_modal:
                init.extend([0.,]*self.nmodes)
                bounds[0].extend([-jnp.inf,]*self.nmodes)
                bounds[1].extend([jnp.inf,]*self.nmodes)
            else:
                init.extend([0.,]*self.model.Np*self.model.Np)
                bounds[0].extend([-jnp.inf,]*self.model.Np*self.model.Np)
                bounds[1].extend([jnp.inf,]*self.model.Np*self.model.Np)
                
        if self.estimate_amplitude:
            if self.amplitude_modal:
                init.extend([1.,]*self.nmodes)
                bounds[0].extend([0,]*self.nmodes)
                bounds[1].extend([jnp.inf,]*self.nmodes)
            else:
                init.extend([1.,]*self.model.Np*self.model.Np)
                bounds[0].extend([0.,]*self.model.Np*self.model.Np)
                bounds[1].extend([jnp.inf,]*self.model.Np*self.model.Np)

        if self.estimate_spectrum:
            init.extend([0,])
            bounds[0].extend([0])
            bounds[1].extend([jnp.inf])

        if self.estimate_bg:
            init.extend([0,])
            bounds[0].extend([0])
            bounds[1].extend([jnp.inf])
            
        if self.modes_tiptilt is not None:
            nparams = len(self.modes_tiptilt)*len(self.forward_model.dz)
            init.extend([0,]*nparams)
            bounds[0].extend([-jnp.inf,]*nparams)
            bounds[1].extend([jnp.inf,]*nparams)

        self.init = jnp.array(init)
        self.bounds = jnp.array(bounds)
         
    def run(self, measured, pupil=None, spectrum=None, weights=None, skip_gauss=False):

        if (not self.estimate_amplitude) and (pupil is None):
            raise ValueError('A pupil transmission function must be supplied when not fitting the amplitude!')
        
        meas = jnp.asarray(measured) / jnp.max(measured,axis=(-2,-1))[:,None,None]
        if weights is None:
            weights = 1/(meas+self.wreg)
        meas = meas - jnp.mean(meas,axis=(-2,-1))[:,None,None]
        
        errf = partial(errfunc, pupil=pupil, meas=meas, wavelengths=self.model.wavelengths,
               fresnel_TFs=self.model.fresnel_TFs, Mx=self.model.Mx, My=self.model.My, weights=weights, Np=self.model.Np,
               modes=self.modes, modes_tiptilt=self.modes_tiptilt, fit_amp=self.estimate_amplitude, fit_spectrum=self.estimate_spectrum,
               fit_bg=self.estimate_bg, spectrum=spectrum)
        valgrad = value_and_grad(errf)
        
        # to do: fix me -- currently hard-coded to phase-only case
        def errf_grad(params, *args, gauss_ph=None):
            val, grad = valgrad(params)
            if self.phase_modal or skip_gauss:
                return val, grad
            else: # pixel-by-pixel fit --> gaussian convolution
                Np = self.model.Np
                grad_sq = gauss_convolve(grad[:Np*Np].reshape((Np,Np)), gauss_ph)
                if (not self.estimate_amplitude) and (not self.estimate_spectrum) and (not self.estimate_bg) and (self.modes_tiptilt is None):
                    grad = grad_sq.flatten()
                else:
                    grad = jnp.concatenate([grad_sq.flatten(), grad[Np*Np:].flatten()],axis=0)
                return val, grad

        solver = self.method(fun=errf_grad, value_and_grad=True, maxiter=self.maxiter, **self.method_kwargs)

        return self._run(solver)
    
    def _run(self, solver, *args, **kwargs):
        gauss_ph = jnp.array(get_gauss(self.gauss_init, (self.model.Np, self.model.Np)))
        state = solver.init_state(self.init, self.bounds, gauss_ph=gauss_ph, *args, **kwargs)
        sol = self.init
        sols, errors, vals = [sol], [state.error], [state.value]
        update = lambda sol,state,gauss_ph: solver.update(sol, state, self.bounds, gauss_ph=gauss_ph, *args, **kwargs)
        jitted_update = jax.jit(update)
        #jitted_update = update
        gauss_vals = jnp.linspace(self.gauss_init, self.gauss_final, num=solver.maxiter)
        for i in range(solver.maxiter):
            
            # to do: figure out how to deal with this gaussian convolution better
            gauss_ph = jnp.array(get_gauss(gauss_vals[i], (self.model.Np, self.model.Np)))
            
            sol, state = jitted_update(sol, state, gauss_ph=gauss_ph)
            sols.append(sol)
            errors.append(state.error)
            vals.append(state.value)
            
        # to do: this returns phase in um (for mysterious scaling reasons) -- fix me!
        return jnp.stack(sols, axis=0), errors, vals

def errfunc(params, pupil, meas, wavelengths, fresnel_TFs, Mx, My, weights, Np, fit_amp=False, fit_spectrum=False, fit_bg=False, modes=None, modes_tiptilt=None, spectrum=None):

    # forward model
    idx = 0
    if modes is None:
        opd_map = params[:Np*Np].reshape((Np,Np)) #gauss_convolve(phi_map.at[fitmask].set(params), gauss)
        idx = Np*Np
    else:
        opd_map = jnp.einsum('i,ijk->jk', params[:len(modes)], modes) #jnp.sum(params[:len(modes),None,None]*modes,axis=0)
        #opd_map = jnp.zeros((Np,Np))
        #for n in range(len(modes)):
        #    opd_map = opd_map + params[n]*modes[n]
        idx = len(modes)
        
    if fit_amp: # eventually handle modal amplitude fit?
        amp = params[idx:idx+Np*Np].reshape(opd_map.shape)
        idx = idx + Np*Np
    else:
        amp = pupil

    if fit_spectrum:
        spec_val = params[idx:idx+1] # not actually a slope
        idx = idx + 1
        spectrum = jnp.linspace(1, spec_val, num=len(wavelengths)) # linear, pinned to 1 at wavelength[0]
    else:
        spectrum = spectrum

    if fit_bg: # not implemented! (and probably shouldn't be)
        bg = params[idx:idx+1]
        idx = idx + 1
    else:
        bg = 0

    if modes_tiptilt is not None:
        tt_params = params[-len(modes_tiptilt)*len(meas):].reshape((len(modes_tiptilt),len(meas)))
        opd_map = opd_map + jnp.einsum('ij,ikl->jkl', tt_params, modes_tiptilt)

    Ik = forward_propagate(amp, opd_map, wavelengths, fresnel_TFs, Mx, My, spectrum=spectrum)
    Ik = Ik - jnp.mean(Ik,axis=(-2,-1))[:,None,None]
    #Ik /= jnp.max(Ik,axis=(-2,-1))[:,None,None]

    # error function
    err = get_err(meas, Ik, weights)
    #err = jnp.sum(weights*(Ik - meas)**2)

    return err

def get_err(Imeas, Imodel, weights):
    '''1 - weighted correlation coefficient (provided that Imeas, Imodel are mean-substracted)'''
    #return jnp.sum(weights*(Imeas - Imodel)**2)
    K = len(weights)
    t1 = jnp.sum(weights * Imodel * Imeas, axis=(-2,-1))**2
    t2 = jnp.sum(weights * Imeas**2, axis=(-2,-1))
    t3 = jnp.sum(weights * Imodel**2, axis=(-2,-1))
    return 1 - 1/K * jnp.sum(t1/(t2*t3), axis=0)
