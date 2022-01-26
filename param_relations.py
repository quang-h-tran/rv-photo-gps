import numpy as np
import pandas as pd
import arviz as az
from math_utils import calc_veq, calc_prot
import matplotlib.pyplot as plt

from scipy.optimize import minimize

N, inc, obl, prot, veq = 25, 85.0, 5.5, 7.2, calc_veq(7.2)
#N, inc, obl, prot, veq = 37, 43.2, 3.2, 12.2, calc_veq(12.2)

def plot_relation(fn, log=True):

    trace = az.from_netcdf(fn)

    names = list(trace.posterior.data_vars)

    rho = trace.posterior['rho'].values.flatten()
    sig = trace.posterior['sigma'].values.flatten()

    if log:
        x = np.log(rho)
        y = np.log(sig)
        x_label = r"log($\rho$)"
        y_label = r"log($\sigma$)"
    else:
        x = rho
        y = sig
        x_label = r"$\rho$"
        y_label = r"$\sigma$"

    # Define the optimization objective
    def f(theta):
        return np.median(np.abs(theta[1]*x + theta[0] - y))

    linear_coeff = np.polyfit(x, y, 1)
    linear = np.poly1d(linear_coeff)

    initial_theta = linear_coeff
    res = minimize(f, initial_theta)

    print(linear_coeff)
    print(res.x)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5),
                           gridspec_kw={'hspace':0.1})

    ax[0].scatter(x, y, s=15, alpha=0.25)

    XX = np.linspace(np.min(x), np.max(x))
    ax[0].plot(XX, linear(XX), lw=2.0, ls='--')
    ax[0].plot(XX, res.x[1]*XX + res.x[0], lw=2.0, ls='--', color='black')

    #ax[1].scatter(np.exp(x), np.exp(y/(res.x[1]*x + res.x[0])), s=15, alpha=0.25)
    #ax[1].scatter(np.exp(x), np.exp(y), s=15, alpha=0.25, color='black')
    #ax[1].scatter(np.exp(x), np.exp(y/linear(x)), s=15, alpha=0.25)
    ax[1].scatter(x, y/(res.x[1]*x + res.x[0]), s=15, alpha=0.25)
    ax[1].scatter(x, y/linear(x), s=15, alpha=0.25, color='black')

    ax[0].set_xlabel(x_label, size=20)
    ax[1].set_xlabel(x_label, size=20)

    ax[1].set_ylim(bottom=0)

    ax[0].set_ylabel(y_label, size=20)
    ax[1].set_ylabel(y_label + r" / $\gamma$", size=20)

    #plt.tight_layout()
    plt.show()

def compare_posteriors(lc_fn, rv_fn):

    lc_c = '#5198E6'
    rv_c = '#F58D2F'

    #names = list(lc_trace.posterior.data_vars)
    #print(names)
    names = ['period', 'rho', 'sigma', 'log_Q0', 'sigma_rot']

    lc_trace = az.from_netcdf(lc_fn)
    rv_trace = az.from_netcdf(rv_fn)

    #lc_summary = az.summary(lc_trace, var_names=names, hdi_prob=0.68)
    #rv_summary = az.summary(rv_trace, var_names=names, hdi_prob=0.68)

    #print(summary)

    #vals = [summary['mean'][n] for n in names]
    #min_e  = [summary['mean'][n] - summary['hdi_16%'][n] for n in names]
    #plus_e = [summary['hdi_84%'][n] - summary['mean'][n] for n in names]

    lc_per = lc_trace.posterior['period'].values.flatten()
    lc_rho = lc_trace.posterior['rho'].values.flatten()
    lc_sig = lc_trace.posterior['sigma'].values.flatten()
    lc_log_Q = lc_trace.posterior['log_Q0'].values.flatten()
    lc_q1 = np.log(0.5 + np.exp(lc_log_Q) + np.exp(lc_log_Q))
    lc_q2 = np.log(0.5 + np.exp(lc_log_Q))
    lc_sig_rot = lc_trace.posterior['sigma_rot'].values.flatten()

    lc_post = [lc_per, lc_rho, lc_sig, lc_sig_rot, lc_q1, lc_q2]

    rv_per = rv_trace.posterior['period'].values.flatten()
    rv_rho = rv_trace.posterior['rho'].values.flatten()
    rv_sig = rv_trace.posterior['sigma'].values.flatten()
    rv_log_Q = rv_trace.posterior['log_Q0'].values.flatten()
    rv_q1 = np.log(0.5 + np.exp(rv_log_Q) + np.exp(rv_log_Q))
    rv_q2 = np.log(0.5 + np.exp(rv_log_Q))
    rv_sig_rot = rv_trace.posterior['sigma_rot'].values.flatten()

    rv_post = [rv_per, rv_rho, rv_sig, rv_sig_rot, rv_q1, rv_q2]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 6),
                           gridspec_kw={'hspace':0.5, 'wspace':0.1})

    ax_idx = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    ax_label = [r'Period', r'$\rho$', r'$\sigma$',
                r'$\sigma_\mathrm{rot}$', r'$Q_1$', r'$Q_2$']

    for i in range(6):
        ax[ax_idx[i]].hist(lc_post[i], 50, density=False, histtype='stepfilled',
                           facecolor=lc_c, alpha=0.35, stacked=True)

        ax[ax_idx[i]].hist(rv_post[i], 50, density=False, histtype='stepfilled',
                           facecolor=rv_c, alpha=0.35, stacked=True)

        ax[ax_idx[i]].set_yticklabels([])
        ax[ax_idx[i]].set_xlabel(ax_label[i])

    plt.tight_layout()
    plt.show()

    #if param == "q1":
    #    f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])) + np.exp(float(f_gp["log_dQ"])))
    #elif param == "q2":
    #    f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])))

lc_posteriors_fn = 'Spots/Limited Time Vector/Orientation 1/Traces/Decay Rate/tau_0.050_lc_trace.nc'
rv_posteriors_fn = 'Spots/Limited Time Vector/Orientation 1/Traces/Decay Rate/tau_0.050_rv_trace.nc'

compare_posteriors(lc_posteriors_fn, rv_posteriors_fn)

#plot_relation(rv_posteriors_fn, log=False)
