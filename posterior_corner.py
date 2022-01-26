import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as st

def get_optimal_bins(s, thres=3):

    def count_zero(data, b=25):
        h_ = np.histogram(data, bins=b)

        bin_zero_count = list(h_[0]).count(0)

        if bin_zero_count > thres:
            return b - 1, bin_zero_count
        else:
            return b, bin_zero_count

    b, bzc = count_zero(s)
    while bzc > thres:
        b, bzc = count_zero(s, b)

    return b

def plot_posteriors(trace, labels=None, fmts=None, fn=None, type='lc', use_corner=False):

    if type == 'lc':
        cmap_colors = 'Blues_r'
    elif type == 'rv':
        cmap_colors = 'Oranges_r'
    else:
        cmap_colors = 'Purples_r'

    names = list(trace.posterior.data_vars)

    names.remove("log_period")
    names.remove("pred")
    names.remove("pred_sigma")
    names.remove("k_val")
    nparams = len(names)

    if labels is None:
        if type == 'lc':
            labels = [r'$\mu_\mathrm{phot}$', r'log(jit)', r'log$(\mathrm{d}Q)$',
                      r'log($\sigma_\mathrm{rot} - \sigma$)',
                      r'$\sigma$', r'$\rho$', r'$\sigma_\mathrm{rot}$',
                      r'$P$', r'log$(Q_0)$', r'$f$']
        elif type == 'rv':
            labels = [r'$\mu_\mathrm{rv}$', r'log(jit)', r'log$(\mathrm{d}Q)$',
                      r'log($\sigma_\mathrm{rot} - \sigma$)',
                      r'$\sigma$', r'$\rho$', r'$\sigma_\mathrm{rot}$',
                      r'$P$', r'log$(Q_0)$', r'$f$']
        elif type == 'lc_prime':
            labels = [r'$\mu_\mathrm{rv}$', r'log$(\mathrm{d}Q)$',
                      r'log($\sigma_\mathrm{rot} - \sigma$)',
                      r'$\sigma$', r'$\rho$', r'$\sigma_\mathrm{rot}$',
                      r'$P$', r'log$(Q_0)$', r'$f$']

    if fmts is None:
        fmts = ["{:.1f}"] * nparams

    chains = [trace.posterior[n].values.flatten() for n in names]

    if use_corner:
        import corner
        import arviz.labels as azl

        del trace.posterior["k_val"]
        del trace.posterior["pred"]
        del trace.posterior["pred_sigma"]
        del trace.posterior["log_period"]
        del trace.posterior["sigma_rot"]

        label_map = {'mean':r'$\mu$', 'log_dQ':r'log$(\mathrm{d}Q)$',
                     'log_sr_m_s':r'log($\sigma_\mathrm{rot} - \sigma$)',
                     'sigma':r'$\sigma$', 'rho':r'$\rho$',
                     'period':r'$P$', 'log_Q0':r'log$(Q_0)$',
                     'log_jitter':r"log(jit)", 'f':r'$f$'}

        labeller = azl.MapLabeller(var_name_map=label_map)

        figure = corner.corner(trace, divergences=True, labeller=labeller)

        #chains = np.vstack(chains)
        #figure = corner.corner(chains.T, labels=labels,
        #               quantiles=[0.16, 0.5, 0.84],
        #               show_titles=True, title_kwargs={"fontsize": 18})

        #ax_list = figure.axes

        #for a in ax_list:
        #    a.tick_params(axis='both', which='major', width=1.5, size=5, labelsize=16)
        #    a.tick_params(axis='both', which='minor', width=1.5, size=2, labelsize=16)

        #    a.tick_params(axis='both', which='both',
        #                  labelbottom=True, labeltop=False,
        #                  labelleft=True, labelright=True)

        #    for s in ['top','bottom','left','right']:
        #        a.spines[s].set_linewidth(2.0)

        if fn is None:
            plt.show()
        else:
            plt.savefig(fn, format='png', bbox_inches='tight', dpi=400)

    else:
        summary = az.summary(trace, var_names=names, hdi_prob=0.68)

        print(summary)

        vals = [summary['mean'][n] for n in names]
        min_e  = [summary['mean'][n] - summary['hdi_16%'][n] for n in names]
        plus_e = [summary['hdi_84%'][n] - summary['mean'][n] for n in names]

        fig  = plt.figure(figsize=(20, 16))

        grid = plt.GridSpec(nparams, nparams, hspace=0.085, wspace=0.085,
                            left=0.075, right=0.975, top=0.975, bottom=0.075)

        diag = [grid[i, i] for i in range(nparams)]

        diag_axes = []
        # Diagonals, histograms
        for n in range(nparams):
            samps = chains[n]

            h = fig.add_subplot(diag[n])
            diag_axes.append(h)
            b = get_optimal_bins(samps)

            h_ = h.hist(samps, bins=b, histtype='step', linewidth=1.5,
                        color='black', rasterized=False)

            h.axvline(vals[n], lw=1.5, color='gray', rasterized=False)

            h.axvline(vals[n]+plus_e[n], ls='--', lw=1.5, color='gray', rasterized=False)
            h.axvline(vals[n]-min_e[n], ls='--', lw=1.5, color='gray', rasterized=False)

            h.tick_params(labelleft=False)
            if n < nparams-1:
                h.tick_params(labelbottom=False)
            else:
                h.tick_params(axis='x', rotation=65)

            h.set_ylim(top=max(h_[0])*1.1)

            title_h = labels[n] + "$ = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(vals[n], min_e[n], plus_e[n])
            h.set_title(title_h, fontsize=16)

        # Do columns
        first_cols = []
        for n in range(nparams):
            m_cols = np.array(range(nparams)) > n
            m_cols = np.array(range(nparams))[m_cols]
            for m in m_cols:
                if n == 0:
                    a = fig.add_subplot(grid[m, n], sharex=diag_axes[n])
                    first_cols.append(a)
                    if m < nparams - 1:
                        a.tick_params(labelbottom=False)
                    a.tick_params(axis='y')
                    a.get_yaxis().get_major_formatter().set_useOffset(False)

        for n in range(nparams):
            m_cols = np.array(range(nparams)) > n
            m_cols = np.array(range(nparams))[m_cols]
            for m in m_cols:
                if n != 0:
                    a = fig.add_subplot(grid[m, n], sharex=diag_axes[n], sharey=first_cols[m-1])
                    a.tick_params(labelleft=False)

                if m < nparams-1 and n != 0:
                    a.tick_params(labelbottom=False)
                else:
                    a.tick_params(axis='x', rotation=65)

        list_of_axes = fig.axes
        grouped_axes = []
        for n in reversed(range(nparams+1)):
            if n != 0:
                grouped_axes.append(list_of_axes[:n])
                list_of_axes = list_of_axes[n:]

        for n in range(nparams):
            if n == nparams-1:
                ax = grouped_axes[0][-1]
                ax.set_xlabel(labels[-1], fontsize=24)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
            else:
                i = n + 1
                ax = grouped_axes[i][-1]
                ax.set_xlabel(labels[n], fontsize=24)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)

        for n in range(len(grouped_axes[1])):
            ax = grouped_axes[1][n]
            ax.set_ylabel(labels[n+1], fontsize=24)

        for g in range(len(grouped_axes)-1):
            g_i = g + 1
            group = grouped_axes[g_i]
            for n in range(len(group)):
                ax = group[n]
                x = chains[g]
                y = chains[g_i:][n]
                xmin, xmax = min(chains[g]), max(chains[g])
                ymin, ymax = min(chains[g_i:][n]), max(chains[g_i:][n])
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = st.gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)

                #cfset = ax.contourf(xx, yy, f, cmap=cmap_colors, levels=6)
                #cset = ax.contour(xx, yy, f, colors='gray', levels=6, alpha=0.5)

                cfset = ax.hist2d(x, y, bins=25, cmap=cmap_colors)#, levels=6)
                #cset = ax.contour([x, y], values, colors='gray', levels=6, alpha=0.5)

        ax_list = fig.axes

        for a in ax_list:
            a.tick_params(axis='both', which='major', width=1.5, size=5, labelsize=20)
            a.tick_params(axis='both', which='minor', width=1.5, size=2, labelsize=20)

            for s in ['top','bottom','left','right']:
                a.spines[s].set_linewidth(2.0)

        if fn is None:
            plt.show()
        else:
            plt.savefig(fn, format='png', bbox_inches='tight', dpi=400)
