from functools import partial

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import value_and_grad

from jaxopt import LBFGS, LBFGSB

from .utils import get_gauss, gauss_convolve
from .model import forward_propagate

class Estimation(object):
    
    def __init__(self, forward_model, estimate_phase=True, estimate_amplitude=False, phase_modal=False, amplitude_modal=False, wreg=1e-2, modes=None, method=LBFGSB, maxiter=100):
        self.forward_model = forward_model
        self.method = method
        self.wreg = wreg
        self.model = forward_model
        
        self.estimate_phase = estimate_phase
        self.estimate_amplitude = estimate_amplitude
        self.phase_modal = phase_modal
        self.amplitude_modal = amplitude_modal
        
        self.modes = modes
        if modes is not None:
            self.nmodes = len(modes)
        else:
            self.nmodes = None
        
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
                
        self.init = jnp.array(init)
        self.bounds = jnp.array(bounds)
         
    def run(self, measured, pupil=None):

        if (not self.estimate_amplitude) and (pupil is None):
            raise ValueError('A pupil transmission function must be supplied when not fitting the amplitude!')
        
        meas = jnp.asarray(measured) / jnp.max(measured,axis=(-2,-1))[:,None,None]
        weights = 1/(meas+self.wreg)
        
        errf = partial(errfunc, pupil=pupil, meas=meas, wavelengths=self.model.wavelengths,
               fresnel_TFs=self.model.fresnel_TFs, Mx=self.model.Mx, My=self.model.My, weights=weights, modes=self.modes, fit_amp=self.estimate_amplitude)
        valgrad = value_and_grad(errf)
        
        # to do: fix me -- currently hard-coded to phase-only case
        def errf_grad(params, gauss_ph):
            val, grad = valgrad(params)
            Np = self.model.Np
            grad_sq = gauss_convolve(grad[:Np*Np].reshape((self.model.Np,self.model.Np)), gauss_ph)
            if not self.estimate_amplitude:
                grad = grad_sq.flatten()
            else:
                grad = jnp.concatenate([grad_sq.flatten(), grad[Np*Np:].flatten()],axis=0)
            return val, grad
        
        solver = self.method(errf_grad, value_and_grad=True, maxiter=self.maxiter, maxls=100, history_size=10, tol=1e-4, verbose=False)

        return self._run(solver)
    
    def _run(self, solver, *args, **kwargs):
        gauss_ph = jnp.array(get_gauss(self.gauss_init, (self.model.Np, self.model.Np)))
        state = solver.init_state(self.init, self.bounds, gauss_ph, *args, **kwargs)
        sol = self.init
        sols, errors = [sol], [state.error]
        update = lambda sol,state,gauss_ph: solver.update(sol, state, self.bounds, gauss_ph=gauss_ph, *args, **kwargs)
        jitted_update = jax.jit(update)
        for i in range(solver.maxiter):
            
            # to do: figure out how to deal with this gaussian convolution better
            gauss_ph = jnp.array(get_gauss(self.gauss_final*i/solver.maxiter + self.gauss_init, (self.model.Np, self.model.Np)))
            
            sol, state = jitted_update(sol, state, gauss_ph=gauss_ph)
            sols.append(sol)
            errors.append(state.error)
            
        # to do: this returns phase in um (for mysterious scaling reasons) -- fix me!
        return jnp.stack(sols, axis=0), errors

def errfunc(params, pupil, meas, wavelengths, fresnel_TFs, Mx, My, weights, fit_amp=False, modes=None):

    # forward model
    Np = len(pupil)
    if modes is None:
        opd_map = params[:Np*Np].reshape(pupil.shape) #gauss_convolve(phi_map.at[fitmask].set(params), gauss)
    else:
        opd_map = jnp.sum(params[:len(modes),None,None]*modes,axis=0)
        
    if fit_amp: # eventually handle modal fit
        amp = params[-Np*Np:].reshape(opd_map.shape)
    else:
        amp = pupil
    
    Ik = forward_propagate(amp, opd_map * 1e-6, wavelengths, fresnel_TFs, Mx, My)
    #Ik = forward_model(pupil, opd_map, wavelengths, mq['fnum'], mq['pix_m'], mq['dz'], mq['Np'], mq['Nf'], model_quantities=mq)
    #return Ik
    
    # error function
    err = get_err(meas, Ik, weights)

    return err

def get_err(Imeas, Imodel, weights):
    #return jnp.sum(weights*(Imeas - Imodel)**2)
    K = len(weights)
    t1 = jnp.sum(weights * Imodel * Imeas, axis=(-2,-1))**2
    t2 = jnp.sum(weights * Imeas**2, axis=(-2,-1))
    t3 = jnp.sum(weights * Imodel**2, axis=(-2,-1))
    return 1 - 1/K * jnp.sum(t1/(t2*t3), axis=0)