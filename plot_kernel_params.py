import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math_utils import calc_veq, calc_prot
import pandas as pd
import glob
import numpy as np
import arviz as az
import os.path
from _utils import create_dirs

################################################################################

N, inc, obl, prot, veq = 25, 88.0, 4.0, 8.0, calc_veq(8.0)
N, inc, obl, prot, veq = 50, 86.5, 6.0, 11.0, calc_veq(11.0)

taus = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

def plot_indivdual_post(param, taus):
    trace_end = "_trace.nc"

    #base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)
    base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)

    param_labels = {"mean":r"Mean", "sigma":r"$\sigma$", "log_Q0":r"log$\:Q_0$",
                    "log_dQ":r"log$\:dQ$", "log_jitter":r"log$\:(\mathrm{jitter})$",
                    "period":r"$P$", "sigma_rot":r"$\sigma_\mathrm{rot}$",
                    "rho":r"$\rho$"}

    param_label = param_labels[param]

    f_vals = np.zeros_like(taus)
    rv_vals = np.zeros_like(taus)
    f_prime_vals = np.zeros_like(taus)

    for i in range(len(taus)):
        base_name = "Decay Rate/" + base_fn + "/tau_{0:.3f}".format(taus[i])
        #base_name = "Decay Rate/" + base_fn + "/tau_{0:.3f}".format(taus[i])
        #base_name = "Spot Radius/" + base_fn + "/radius_{0:.2f}".format(rad)

        f_post = 'Traces/' + base_name + '_lc' + trace_end
        f_post = az.from_netcdf(f_post)

        rv_post = 'Traces/' + base_name + '_rv' + trace_end
        rv_post = az.from_netcdf(rv_post)

        f_prime_post = 'Traces/' + base_name + '_lc_prime' + trace_end
        f_prime_post = az.from_netcdf(f_prime_post)

        f_chains = f_post.posterior[param].values.flatten()
        rv_chains = rv_post.posterior[param].values.flatten()
        f_prime_chains = f_prime_post.posterior[param].values.flatten()

        f_vals[i] = np.median(f_chains)
        rv_vals[i] = np.median(rv_chains)
        f_prime_vals[i] = np.median(f_prime_chains)

    fig, (f_ax, f_prime_ax, rv_ax) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True,
                           gridspec_kw={'hspace':0.0})

    lc_c = '#5198E6'
    lc_prime_c = '#8000DC'
    rv_c = '#F58D2F'

    taus = 1. / np.array(taus)

    f_ax.plot(taus, f_vals, '-o', color=lc_c, markersize=6, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)

    f_prime_ax.plot(taus, rv_vals, '-s', color=lc_prime_c, markersize=6, linewidth=1.75, mfc='white',
                    mec=lc_prime_c, mew=1.0)

    rv_ax.plot(taus, f_prime_vals, '-^', color=rv_c, markersize=6, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)

    #rv_ax.set_xlabel(r"$\tau_\mathrm{Decay}$ (days)")
    rv_ax.set_xlabel(r"$\tau_\mathrm{Decay}$ (days)")
    rv_ax.set_xscale('log')

    f_ax.set_ylabel(param_labels[param]+r"$_F$")
    f_prime_ax.set_ylabel(param_labels[param]+r"$_{F^\prime}$")
    rv_ax.set_ylabel(param_labels[param]+r"$_\mathrm{RV}$")

    plt.savefig("Plots/Kernel Parameter Comparison/"+param+"_v_tau.png", dpi=200, bbox_inches='tight')

#params = ["mean", "sigma", "log_dQ", "log_Q0",
#          "log_jitter", "period", "sigma_rot", "rho"]
#
#for p in params:
#     plot_indivdual_post(p, taus)

def plot_individual_MAP(param, var, base_name="Spot Radius"):
    base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)

    base_path = base_name + "/" + base_fn
    fn = "Optimal Kernel Parameters/" + base_path

    param_labels = {"mean":r"Mean", "sigma":r"$\sigma$", "log_Q0":r"log$\:(Q_0)$",
                    "log_dQ":r"log$\:(dQ)$", "log_jitter":r"log$\:(\mathrm{jitter})$",
                    "period":r"$P$ (days)", "sigma_rot":r"$\sigma_\mathrm{rot}$",
                    "rho":r"$\rho$", "q1":r"log$\:Q_1$", "q2":r"log$\:Q_2$", "f":r"f"}
    #param_label = param_labels[param]

    all_f_params = glob.glob(fn + "/*lc.csv")
    all_f_params.sort()
    all_rv_params = glob.glob(fn + "/*rv.csv")
    all_rv_params.sort()
    #all_f_prime_params = glob.glob(fn + "/*lc_prime.csv")
    #all_f_prime_params.sort()

    f_vals = np.zeros(len(all_f_params))
    rv_vals = np.zeros(len(all_rv_params))
    #f_prime_vals = np.zeros(len(all_f_prime_params))

    for i in range(len(all_f_params)):
        f = all_f_params[i]
        f_gp = pd.read_csv(f)
        if param == "q1":
            f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])) + np.exp(float(f_gp["log_dQ"])))
        elif param == "q2":
            f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])))
        else:
            f_vals[i] = float(f_gp[param])
    for j in range(len(all_rv_params)):
        rv = all_rv_params[j]
        rv_gp = pd.read_csv(rv)
        if param == "q1":
            rv_vals[j] = np.log(0.5 + np.exp(float(rv_gp["log_Q0"])) + np.exp(float(rv_gp["log_dQ"])))
        elif param == "q2":
            rv_vals[j] = np.log(0.5 + np.exp(float(rv_gp["log_Q0"])))
        else:
            rv_vals[j] = float(rv_gp[param])
    #for k in range(len(all_f_prime_params)):
    #    fp = all_f_prime_params[k]
    #    f_prime_gp = pd.read_csv(fp)
    #    f_prime_vals[k] = float(f_prime_gp[param])

    lc_c = '#5198E6'
    #lc_prime_c = '#8000DC'
    rv_c = '#F58D2F'

    print(f_vals)
    print("")

    if param != 'period':
        #fig, (f_ax, f_prime_ax, rv_ax) = plt.subplots(nrows=3, ncols=1, figsize=(10, 4), sharex=True,
        fig, (f_ax, rv_ax) = plt.subplots(nrows=2, ncols=1, figsize=(10, 4), sharex=True,
                               gridspec_kw={'hspace':0.0})

        base_labels = {'Spot Radius':r"Spot Radius $(\%)$",
                        'Spot Contrast':r"Spot Contrast $(\%)$",
                        'Decay Rate':r"$\tau_\mathrm{Decay}$ (days)",
                        'Spot Radius and Contrast':r"Radius and Contrast $(\%)$"}

        rv_ax.set_xlabel(base_labels[base_name])
        if base_name == 'Decay Rate':
            rv_ax.set_xscale('log')
            var = 1. / np.array(var)
        else:
            var = np.array(var) * 100.

        f_ax.set_ylabel(param_labels[param]+r"$_F$", size=18)
        #f_prime_ax.set_ylabel(param_labels[param]+r"$_{F^\prime}$", size=18)
        rv_ax.set_ylabel(param_labels[param]+r"$_\mathrm{RV}$", size=18)

        f_ax.plot(var, f_vals, '-o', color=lc_c, markersize=6, linewidth=1.75, mfc='white',
                        mec=lc_c, mew=1.0, label=r"$F$")
        rv_ax.plot(var, rv_vals, '-s', color=rv_c, markersize=6, linewidth=1.75, mfc='white',
                        mec=rv_c, mew=1.0, label=r"RV")
        #f_prime_ax.plot(var, f_prime_vals, '-^', color=lc_prime_c, markersize=6, linewidth=1.75, mfc='white',
        #                mec=lc_prime_c, mew=1.0, label=r"$F^\prime$")

        #for a in (f_ax, f_prime_ax, rv_ax):
        for a in (f_ax, rv_ax):
            a.tick_params(axis='y', which='major', labelsize=16)
            ymin, ymax = a.get_ylim()
            yrange = np.abs(ymax - ymin)
            a.set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
            a.legend(markerscale=1.0, fontsize=16)
    else:
        fig, ax = plt.subplots(figsize=(10, 3))

        base_labels = {'Spot Radius':r"Spot Radius $(\Delta \%)$",
                        'Spot Contrast':r"Spot Contrast (\%)",
                        'Decay Rate':r"$\tau_\mathrm{Decay}$ (days)",
                        'Spot Radius and Contrast':r"Radius and Contrast $(\%)$"}

        ax.set_xlabel(base_labels[base_name])
        if base_name == 'Decay Rate':
            ax.set_xscale('log')
            var = 1. / np.array(var)
        else:
            var = np.array(var) * 100.

        ax.set_ylabel(param_labels[param])

        ax.plot(var, f_vals, '-o', color=lc_c, markersize=6, linewidth=1.75, mfc='white',
                        mec=lc_c, mew=1.0, label=r"$F$")
        ax.plot(var, rv_vals, '-s', color=rv_c, markersize=6, linewidth=1.75, mfc='white',
                        mec=rv_c, mew=1.0, label=r"RV")
        #ax.plot(var, f_prime_vals, '-^', color=lc_prime_c, markersize=6, linewidth=1.75, mfc='white',
        #                mec=lc_prime_c, mew=1.0, label=r"$F^\prime$")

        ymin, ymax = ax.get_ylim()
        yrange = np.abs(ymax - ymin)
        ax.set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])

        ax.legend(markerscale=1.0, fontsize=16)

    plt.savefig("Plots/Kernel Parameter Comparison/"+base_name+"/"+base_fn+"/"+param+".png", dpi=400, bbox_inches='tight')

def compare_MAP(param, var, spot_param="Decay Rate", N_ori=[1]):

    base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)

    param_labels = {"mean":r"Mean", "sigma":r"$\sigma$", "log_Q0":r"log$\:(Q_0)$",
                    "log_dQ":r"log$\:(dQ)$", "log_jitter":r"log$\:(\mathrm{jitter})$",
                    "period":r"$P$ (days)", "sigma_rot":r"$\sigma_\mathrm{rot}$",
                    "rho":r"$\rho$", "q1":r"log$\:Q_1$", "q2":r"log$\:Q_2$", "f":r"f"}

    if spot_param == "Decay Rate":
        var_name = "tau_"
    if spot_param == "Spot Radius":
        var_name = "rad_"
    if spot_param == "Spot Contrast":
        var_name = "con_"
    if spot_param == "Spot Radius and Contrast":
        var_name = ""

    lc_c = '#5198E6'
    rv_c = '#F58D2F'

    if param != 'period':
        fig, (f_ax, rv_ax) = plt.subplots(nrows=2, ncols=1, figsize=(8, 3), sharex=True,
                               gridspec_kw={'hspace':0.0})

        base_labels = {'Spot Radius':r"Spot Radius $(\%)$",
                        'Spot Contrast':r"Spot Contrast $(\%)$",
                        'Decay Rate':r"$\tau_\mathrm{Decay}$ (days)",
                        'Spot Radius and Contrast':r"Radius and Contrast $(\%)$"}

        rv_ax.set_xlabel(base_labels[spot_param])
        if spot_param == 'Decay Rate':
            rv_ax.set_xscale('log')
            var = 1. / np.array(var)
        else:
            var = np.array(var) * 100.

        f_ax.set_ylabel(param_labels[param]+r"$_F$", size=18)
        rv_ax.set_ylabel(param_labels[param]+r"$_\mathrm{RV}$", size=18)
    else:
        fig, ax = plt.subplots(figsize=(10, 3))

        base_labels = {'Spot Radius':r"Spot Radius $(\Delta \%)$",
                        'Spot Contrast':r"Spot Contrast (\%)",
                        'Decay Rate':r"$\tau_\mathrm{Decay}$ (days)",
                        'Spot Radius and Contrast':r"Radius and Contrast $(\%)$"}

        ax.set_xlabel(base_labels[spot_param])
        if spot_param == 'Decay Rate':
            ax.set_xscale('log')
            var = 1. / np.array(var)
        else:
            var = np.array(var) * 100.

        ax.set_ylabel(param_labels[param])

    for n in N_ori:
        paths = create_dirs(n, N, inc, obl, prot, veq)
        spot_param = spot_param + "/"

        var_paths = [i+spot_param for i in paths]

        op_kern_param_path = var_paths[1]

        all_f_params = glob.glob(op_kern_param_path + "/*lc.csv")
        all_f_params.sort()
        all_rv_params = glob.glob(op_kern_param_path + "/*rv.csv")
        all_rv_params.sort()

        f_vals = np.zeros(len(all_f_params))
        rv_vals = np.zeros(len(all_rv_params))

        for i in range(len(all_f_params)):
            f = all_f_params[i]
            f_gp = pd.read_csv(f)
            if param == "q1":
                f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])) + np.exp(float(f_gp["log_dQ"])))
            elif param == "q2":
                f_vals[i] = np.log(0.5 + np.exp(float(f_gp["log_Q0"])))
            else:
                f_vals[i] = float(f_gp[param])

        for j in range(len(all_rv_params)):
            rv = all_rv_params[j]
            rv_gp = pd.read_csv(rv)
            if param == "q1":
                rv_vals[j] = np.log(0.5 + np.exp(float(rv_gp["log_Q0"])) + np.exp(float(rv_gp["log_dQ"])))
            elif param == "q2":
                rv_vals[j] = np.log(0.5 + np.exp(float(rv_gp["log_Q0"])))
            else:
                rv_vals[j] = float(rv_gp[param])

        if param != 'period':
            f_ax.plot(var, f_vals, '-', color=lc_c, markersize=2, linewidth=2.0, mfc='white',
                            mec=lc_c, mew=1.0, label=r"$F$", alpha=0.25)
            rv_ax.plot(var, rv_vals, '-', color=rv_c, markersize=2, linewidth=2.0, mfc='white',
                            mec=rv_c, mew=1.0, label=r"RV", alpha=0.25)
        else:
            ax.plot(var, f_vals, '-', color=lc_c, markersize=2, linewidth=2.0, mfc='white',
                            mec=lc_c, mew=1.0, label=r"$F$", alpha=0.25)
            ax.plot(var, rv_vals, '-', color=rv_c, markersize=2, linewidth=2.0, mfc='white',
                            mec=rv_c, mew=1.0, label=r"RV", alpha=0.25)

    if param != 'period':
        for a in (f_ax, rv_ax):
            a.tick_params(axis='y', which='major', labelsize=16)
            ymin, ymax = a.get_ylim()
            yrange = np.abs(ymax - ymin)
            a.set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
            a.set_xlim([np.min(var), np.max(var)])
            #a.set_yscale('log')

            handles, labels = a.get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            a.legend(handles, labels, markerscale=1.0, fontsize=16)
    else:
        ymin, ymax = ax.get_ylim()
        yrange = np.abs(ymax - ymin)
        ax.set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
        ax.set_xlim([np.min(var), np.max(var)])

        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax.legend(handles, labels, markerscale=1.0, fontsize=16)

    plt.savefig("Plots/Kernel Parameter Comparison/Decay Rate/"+base_fn+"/"+param+".png", dpi=400, bbox_inches='tight')
    #plt.tight_layout()
    #plt.show()

#for n in [20, 21]:

taus = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
taus.sort()


vars = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 2.0]

params = ["sigma", "log_dQ", "log_Q0", "q1", "q2",
          "period", "sigma_rot", "rho", "f"]
for p in params:
    compare_MAP(p, taus, N_ori=np.arange(2, 21, 1))

#for p in params:
    #plot_individual_MAP(p, taus, base_name='Decay Rate')
    #plot_individual_MAP(p, vars, base_name='Spot Radius and Contrast')
    #plot_individual_MAP(p, vars, base_name='Spot Radius')
    #plot_individual_MAP(p, vars, base_name='Spot Contrast')

# for tau only
def plot_all_MAP():

    all_lc_params = glob.glob(base_fn + "/*lc.csv")
    all_lc_params.sort()
    all_rv_params = glob.glob(base_fn + "/*rv.csv")
    all_rv_params.sort()

    tau = np.zeros(len(all_lc_params))

    lc_mean, rv_mean = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_sigma, rv_sigma = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_log_dQ, rv_log_dQ = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_log_Q0, rv_log_Q0 = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_log_jitter, rv_log_jitter = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_period, rv_period = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_sigma_rot, rv_sigma_rot = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_log_sr_m_s, rv_log_sr_m_s = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))
    lc_rho, rv_rho = np.zeros(len(all_lc_params)), np.zeros(len(all_rv_params))

    #base_fn = 'Optimal Kernel Parameters/Decay Rate/N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)

    for i in range(len(all_lc_params)):
        f = all_lc_params[i]
        r = all_rv_params[i]
        t = float(f.split("tau")[1][1:6])
        lc_params, rv_params = pd.read_csv(f), pd.read_csv(r)

        tau[i] = t
        lc_mean[i], rv_mean[i] = float(lc_params["mean"]), float(rv_params["mean"])
        lc_sigma[i], rv_sigma[i] = float(lc_params["sigma"]), float(rv_params["sigma"])
        lc_log_dQ[i], rv_log_dQ[i] = float(lc_params["log_dQ"]), float(rv_params["log_dQ"])
        lc_log_Q0[i], rv_log_Q0[i] = float(lc_params["log_Q0"]), float(rv_params["log_Q0"])
        lc_log_jitter[i], rv_log_jitter[i] = float(lc_params["log_jitter"]), float(rv_params["log_jitter"])
        lc_period[i], rv_period[i] = float(lc_params["period"]), float(rv_params["period"])
        lc_sigma_rot[i], rv_sigma_rot[i] = float(lc_params["sigma_rot"]), float(rv_params["sigma_rot"])
        lc_log_sr_m_s[i], rv_log_sr_m_s[i] = float(lc_params["log_sr_m_s"]), float(rv_params["log_sr_m_s"])
        lc_rho[i], rv_rho[i] = float(lc_params["rho"]), float(rv_params["rho"])

    tau = 1. / tau

    # Set colors
    lc_c = '#5198E6'
    rv_c = '#F58D2F'

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(22, 10),
                           gridspec_kw={'hspace':0.5, 'wspace':0.25})

    # mean
    ax[0,0].plot(tau, lc_mean, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[0,0].plot(tau, rv_mean, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[0,0].set_ylabel(r"Mean")

    # sigma
    ax[0,1].plot(tau, lc_sigma/np.nanmin(lc_sigma), '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[0,1].plot(tau, rv_sigma/np.nanmin(rv_sigma), '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[0,1].set_ylabel(r"$\sigma / \sigma_\mathrm{min}$")

    # log_dQ
    ax[0,2].plot(tau, lc_log_dQ, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[0,2].plot(tau, rv_log_dQ, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[0,2].set_ylabel(r"log$\:dQ$")

    # log_Q0
    ax[1,0].plot(tau, lc_log_Q0, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[1,0].plot(tau, rv_log_Q0, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[1,0].set_ylabel(r"log$\:Q_0$")

    # log_jitter
    ax[1,1].plot(tau, lc_log_jitter, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[1,1].plot(tau, rv_log_jitter, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[1,1].set_ylabel(r"log$\:(\mathrm{jitter})$")

    # log_period
    ax[1,2].plot(tau, lc_period, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[1,2].plot(tau, rv_period, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[1,2].set_ylabel(r"$P$")

    # sigma_rot
    ax[2,0].plot(tau, lc_sigma_rot, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[2,0].plot(tau, rv_sigma_rot, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[2,0].set_ylabel(r"$\sigma_\mathrm{rot}$")

    # log_sr_m_s
    ax[2,1].plot(tau, lc_log_sr_m_s, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[2,1].plot(tau, rv_log_sr_m_s, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[2,1].set_ylabel(r"log($\:\sigma_\mathrm{rot}-\sigma$)")

    # rho
    ax[2,2].plot(tau, lc_rho, '-o', color=lc_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=lc_c, mew=1.0)
    ax[2,2].plot(tau, rv_rho, '-s', color=rv_c, markersize=3, linewidth=1.75, mfc='white',
                    mec=rv_c, mew=1.0)
    ax[2,2].set_ylabel(r"$\rho$")

    for a in ax:
        for ai in a:
            ai.set_xlabel(r"$\tau_\mathrm{Decay}$ (days)")
            ai.set_xscale('log')

    plt.savefig("params_v_tau.png", dpi=200)

#plot_all_MAP()
