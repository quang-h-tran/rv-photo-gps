import scipy.stats as stats
import numpy as np
import starry
import matplotlib.pyplot as plt

var_path = "example data/tau_0.050"

t_flux = np.load(var_path + "_t.npy")#[:1000]
flux = np.load(var_path + "_f.npy")#[:1000]
flux_err = np.load(var_path + "_ferr.npy")#[:1000]

mu = np.mean(flux)
flux = (flux / mu - 1) * 1e3
flux_err = flux_err * 1e3 / mu

t_rad = np.load(var_path + "_t.npy")#[250:1750]
rv = np.load(var_path + "_rv.npy")#[250:1750]
rv_err = np.load(var_path + "_rverr.npy")#[250:1750]


fig, ax = plt.subplots(nrows=2, figsize=(14, 6), sharex=True)

#ax[0].scatter(t_flux, flux, c='black', marker='o', s=5.0, label='LC', zorder=-1)
#ax[1].scatter(t_rad, rv, c='black', marker='o', s=5.0, label='RV', zorder=-1)

ax[0].plot(t_flux, flux-flux_err, color='C0', lw=1.5, label='True LC', zorder=10)
ax[1].plot(t_rad, rv-rv_err, color='C0', lw=1.5, label='True RV', zorder=10)

ax[0].scatter(t_flux, flux, c='black', marker='o', s=5.0, label='LC Data', zorder=-1)
ax[1].scatter(t_rad, rv, c='black', marker='o', s=5.0, label='RV Data', zorder=-1)

ax[0].set_ylabel(r"Relative Flux (ppt)", fontsize=16)
ax[1].set_ylabel(r"Radial Velocity (m s$^{-1}$)", fontsize=16)

ax[1].set_xlabel(r"Time (days)")

ax[0].set_xlim(t_flux.min(), t_flux.max())

ax[0].legend(fontsize=14, markerscale=2.0)
ax[1].legend(fontsize=14, markerscale=2.0)

plt.savefig("Plots/example_data.png", bbox_inches='tight', dpi=400)

#plt.show()
