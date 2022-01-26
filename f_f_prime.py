import exoplanet as xo

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import arviz as az
import os
from _utils import create_dirs
from scipy import optimize

import pandas as pd
import numpy as np
import lightkurve as lk

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from math_utils import calc_veq, calc_prot

################################################################################

def ready_data(y, yerr):
    yerr = np.abs(yerr)

    mu = np.mean(y)
    y = (y / mu - 1) * 1e3
    yerr = yerr * 1e3 / mu

    return y, yerr

def calc_lsc(x, y):

    results = xo.estimators.lomb_scargle_estimator(
        x, y, max_peaks=1, min_period=0.5, max_period=20.0, samples_per_peak=50
    )

    peak = results["peaks"][0]
    freq, power = results["periodogram"]

    return peak, freq, power

def gp_lc(x, y, yerr):

    peak, freq, power = calc_lsc(x, y)

    with pm.Model() as model:

        # The mean value of the time series
        mean = pm.Normal("mean", mu=0.0, sigma=1.0)

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=1.0)

        # A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0),
            testval=1.5)
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(1.0, 20.0))

        #sigma_rot = pm.InverseGamma(
        #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(0.5, 5.0),
        #    testval=3.0)
        log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=0.75)
        period = pm.Deterministic("period", tt.exp(log_period))
        log_Q0 = pm.HalfNormal("log_Q0", sigma=1.0)
        log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=1.0)
        f = pm.Uniform("f", lower=0.01, upper=2.5)

        # This forces sigma_rot > sigma
        log_sr_m_s = pm.Normal("log_sr_m_s", mu=0.0, sigma=1.0)
        sigma_rot = pm.Deterministic("sigma_rot", tt.exp(log_sr_m_s)+sigma)

        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0/3.0)
        kernel += terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,)

        gp = GaussianProcess(
            kernel,
            t=x,
            diag=yerr ** 2 + tt.exp(2 * log_jitter),
            mean=mean,
            quiet=True,)

        gp.marginal("gp", observed=y)

        # Compute the mean and sigma of model prediction for plotting purposes
        #mu, variance = gp.predict(y, return_var=True)
        #pm.Deterministic("pred", mu)
        #pm.Deterministic("pred_sigma", tt.sqrt(variance))

        # evaluate kernel values at tau
        #k_val = kernel.get_value(x - x[0])
        #pm.Deterministic("k_val", k_val)

        map_soln = model.test_point

        map_soln = pmx.optimize(map_soln, vars=[log_Q0])
        map_soln = pmx.optimize(map_soln, vars=[log_dQ])
        map_soln = pmx.optimize(map_soln, vars=[f])
        map_soln = pmx.optimize(map_soln, vars=[rho])
        map_soln = pmx.optimize(map_soln, vars=[f, rho, log_dQ, log_Q0])
        map_soln = pmx.optimize(map_soln, vars=[log_period])
        map_soln = pmx.optimize(map_soln, vars=[sigma])
        #map_soln = pmx.optimize(map_soln, vars=[sigma_rot])
        map_soln = pmx.optimize(map_soln, vars=[log_sr_m_s])
        map_soln = pmx.optimize(map_soln, vars=[log_period, sigma, log_sr_m_s])
        map_soln = pmx.optimize(map_soln, vars=[mean, log_jitter])

        #opt_params = map_soln.copy()
        #del opt_params['pred']
        #del opt_params['pred_sigma']
        #del opt_params['k_val']

        #opt_params = pd.DataFrame(opt_params, index=[0])
        #opt_params.to_csv(kern_fn + "_lc.csv",
        #                  index=False)

        # New, optimized kernel
        opt_k = terms.SHOTerm(sigma=map_soln["sigma"], rho=map_soln["rho"], Q=1.0/3.0)
        opt_k += terms.RotationTerm(
            sigma=map_soln["sigma_rot"],
            period=tt.exp(map_soln["log_period"]),
            Q0=tt.exp(map_soln["log_Q0"]),
            dQ=tt.exp(map_soln["log_dQ"]),
            f=map_soln["f"],)

        opt_gp = GaussianProcess(
            opt_k,
            t=x,
            diag=yerr ** 2 + tt.exp(2 * map_soln["log_jitter"]),
            mean=map_soln["mean"],
            quiet=True,)

        return opt_gp

t = np.load('Spots/Orientation 1/Simulated Data/Decay Rate/tau_0.020_t.npy')
f = np.load('Spots/Orientation 1/Simulated Data/Decay Rate/tau_0.020_f_no_err.npy')
f_err = np.load('Spots/Orientation 1/Simulated Data/Decay Rate/tau_0.020_ferr.npy')
rv = np.load('Spots/Orientation 1/Simulated Data/Decay Rate/tau_0.020_rv.npy')
rv_err = np.load('Spots/Orientation 1/Simulated Data/Decay Rate/tau_0.020_rverr.npy')
#rv = rv - rv_err

random = np.random.default_rng(4321)
idx_rvs = np.sort(random.choice(np.arange(len(rv)), size=50, replace=False))

def ffprime_linear(T, F, RV, RV_i, plot=False):

    F_prime = np.gradient(F, T)

     # Build the design matrix and fit for the best-fit coefficients
    A = np.stack((F, F ** 2, F_prime, F * F_prime, np.ones_like(F)), axis=-1)
    w = np.linalg.lstsq(A[RV_i], RV[RV_i], rcond=None)[0]

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6), sharex=True,
                                            gridspec_kw={'hspace':0.0})
        ax1.plot(T, F)
        ax1.set_ylabel("$F$(t)")

        ax2.plot(T, F_prime)
        ax2.set_ylabel("$F'$(t)")

        ax3.plot(T, RV, label="Simulated RV")
        ax3.plot(T, np.dot(A, w), label="Best Fit $FF' Model$")
        ax3.legend(fontsize=14)
        ax3.set_ylabel("RV m s$^{-1}$")
        ax3.set_xlabel("$t$ (days)")
        ax3.set_xlim(t.min(), t.max())

        plt.tight_layout()
        plt.show()

    return np.dot(A, w)

def ffprime_gp(time, flux, flux_err, rvs, rv_ind, plot=False):

    psi = gp_lc(time, flux, np.abs(flux_err))
    psi_predict = psi.predict(flux, t=time).eval()

    #F = 1.0 - psi_predict
    F = psi_predict
    #psi_predict_prime = -np.gradient(psi_predict, time[rv_ind])
    #F_prime = psi_predict_prime
    #F_prime = -np.gradient(psi_predict, time)
    F_prime = np.gradient(psi_predict, time)

    # Build the design matrix and fit for the best-fit coefficients
    A = np.stack((F, F ** 2, F_prime, F * F_prime, np.ones_like(F)), axis=-1)
    w = np.linalg.lstsq(A[rv_ind], rvs[rv_ind], rcond=None)[0]

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6), sharex=True,
                                            gridspec_kw={'hspace':0.0})
        ax1.plot(time, F)
        ax1.set_ylabel("$F$(t)")

        ax2.plot(time, F_prime)
        ax2.set_ylabel("$F'$(t)")

        ax3.plot(time, rvs, label="Simulated RV")
        ax3.plot(time, np.dot(A, w), label="Best Fit $FF' Model$")
        ax3.legend(fontsize=14)
        ax3.set_ylabel("RV m s$^{-1}$")
        ax3.set_xlabel("$t$ (days)")
        ax3.set_xlim(time.min(), time.max())

        plt.tight_layout()
        plt.show()

    return np.dot(A, w)

def rv_rot(f_data, psi_0, R_star=1.0):

    flux, flux_prime = f_data

    little_f = (psi_0 - np.nanmin(flux)) / psi_0

    return - (flux_prime / psi_0) * (1. - (flux / psi_0)) * (R_star / little_f)

def rv_conv(f_data, psi_0):

    flux, flux_prime = f_data

    little_f = (psi_0 - np.nanmin(flux)) / psi_0

    return (1. - (flux / psi_0))**2  / little_f

def rv_act(f_data, A, B, psi_0):

    return (A*rv_rot(f_data, psi_0)) + (B*rv_conv(f_data, psi_0))

def residual(theta, RVs, f_data, rv_ind):
    return RVs[rv_ind] - rv_act(f_data, *theta)[rv_ind]

def ffprime_nonlinear(time, flux, rvs, rv_ind, plot=False):

    flux_prime = np.gradient(flux, time)

    theta0 = [10.0, 10.0, np.nanmax(f[rv_ind])]
    popt, pcov = optimize.leastsq(residual, theta0, args=(rvs, [flux, flux_prime], rv_ind))

    print(popt)

    if plot:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(t, rv, label="Simulated RV")
        ax.plot(t, rv_act([f, f_p], *popt))
        ax.plot(t[idx_rvs], rv[idx_rvs], color='black', marker='o', ls='none', label="Sampled RV")

        ax.set_ylabel("RV (m s$^{-1}$)")
        ax.set_ylim([-375, 410])
        ax.set_xlabel("$t$ (day)")
        ax.set_xlim(t.min(), t.max())

        ax.legend(fontsize=16, markerscale=1.0)

        plt.savefig('Plots/ffprime_nonlinear_model.png', bbox_inches='tight', dpi=400)

    return rv_act([flux, flux_prime], *popt)

def plot_models(time, flux, flux_err, rvs, rv_ind, rv_errs):

    ffprime_linear_model = ffprime_linear(time, flux, rvs, rv_ind)
    ffprime_gp_model = ffprime_gp(time, flux, flux_err, rvs, rv_ind)
    ffprime_nonlinear_model = ffprime_nonlinear(time, flux, rvs, rv_ind)

    fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True,
                                  gridspec_kw={'hspace':0.0, 'height_ratios':[2,1]})

    ax.plot(time, rvs, color='grey', ls='--', zorder=-2)
    ax.plot(time, rvs-rv_errs, color='black', ls='--', label="Simulated RV", zorder=-1)

    if len(rv_ind) < len(time):
        ax.plot(time[rv_ind], rvs[rv_ind], color='black', marker='o', ls='none', zorder=2, label="Sampled RV")
        ax_resid.scatter(time[rv_ind], rvs[rv_ind] - ffprime_linear_model[rv_ind],
                         edgecolors='blue', marker='o', c='none', zorder=10,
                         linewidths=1.0)
        ax_resid.scatter(time[rv_ind], rvs[rv_ind] - ffprime_gp_model[rv_ind],
                         edgecolors='orange', marker='o', c='none', zorder=10,
                         linewidths=1.0)
        ax_resid.scatter(time[rv_ind], rvs[rv_ind] - ffprime_nonlinear_model[rv_ind],
                         edgecolors='pink', marker='o', c='none', zorder=10,
                         linewidths=1.0)

    ax.plot(time, ffprime_linear_model, color='blue', alpha=0.5,
            label="Flexible $FF'$ from Data", lw=2.0, zorder=4)
    ax.plot(time, ffprime_gp_model, color='orange', alpha=0.5,
            label="Flexible $FF'$ from GP", lw=2.0, zorder=5)
    ax.plot(time, ffprime_nonlinear_model, color='pink', alpha=0.5,
            label="Original $FF'$ from Data", lw=2.0, zorder=5)

    ax_resid.plot(time, rvs-rv_errs-ffprime_linear_model, color='blue', alpha=0.35,
                  lw=0.75, zorder=5)
    ax_resid.plot(time, rvs-rv_errs-ffprime_gp_model, color='orange', alpha=0.35,
                  lw=0.75, zorder=5)
    ax_resid.plot(time, rvs-rv_errs-ffprime_nonlinear_model, color='pink', alpha=0.35,
                  lw=0.75, zorder=5)

    ax_resid.axhline(y=0.0, ls='--', lw=2.0, color='black')

    ax.legend(fontsize=12, markerscale=1.0)
    ax.set_ylabel("RV (m s$^{-1}$)")
    ax_resid.set_xlabel("Time (days)")
    ax_resid.set_ylabel("RV Residuals", fontsize=16)

    ax.set_xlim(time.min(), time.max())
    ax.set_ylim([-375, 410])
    ax_resid.set_ylim([-225, 225])

    #plt.savefig('Plots/ffprime_gp_models_sampled_data.png', bbox_inches='tight', dpi=400)
    #plt.savefig('Plots/ffprime_gp_models_full_data.png', bbox_inches='tight', dpi=400)
    plt.tight_layout()
    plt.show()

#plot_models(t, f, f_err, rv, np.arange(len(t)), rv_errs=rv_err)

#random = np.random.default_rng(4321)
#idx_rvs = np.sort(random.choice(np.arange(len(rv)), size=50, replace=False))
#plot_models(t, f, f_err, rv, idx_rvs, rv_errs=rv_err)

################################################################################
