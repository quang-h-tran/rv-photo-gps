import numpy as np
import matplotlib.pyplot as plt
from starry._plotting import get_ortho_latitude_lines, get_ortho_longitude_lines

# Plotting functions
def grid_lines(grid_ax, theta, inc, obl):
    # code taken from starry github, with minor changes to fit situation
    borders = []
    # Anti-aliasing at the edges
    x = np.linspace(-1, 1, 10000)
    y = np.sqrt(1 - x ** 2)
    borders += [grid_ax.fill_between(x, 1.1 * y, y, color="w", zorder=-1)]
    borders += [grid_ax.fill_betweenx(x, 1.1 * y, y, color="w", zorder=-1)]
    borders += [grid_ax.fill_between(x, -1.1 * y, -y, color="w", zorder=-1)]
    borders += [grid_ax.fill_betweenx(x, -1.1 * y, -y, color="w", zorder=-1)]
    borders += grid_ax.plot(x, y, "k-", alpha=1, lw=1.5, zorder=0)
    borders += grid_ax.plot(x, -y, "k-", alpha=1, lw=1.5, zorder=0)

    lats = get_ortho_latitude_lines(inc=np.deg2rad(inc), obl=np.deg2rad(obl))
    latlines = [None for n in lats]
    for n, l in enumerate(lats):
        (latlines[n],) = grid_ax.plot(l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=0)
    lons = get_ortho_longitude_lines(inc=np.deg2rad(inc), obl=np.deg2rad(obl),
                                     theta=np.deg2rad(theta[0]))
    lonlines = [None for n in lons]
    for n, l in enumerate(lons):
        (lonlines[n],) = grid_ax.plot(l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=0)

    return lonlines, latlines, borders

def plot_animation(t, f, theta, image, N, inc, obl, ydeg, veq, fn=None):

    from matplotlib import animation

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 3),
                           gridspec_kw={'hspace':0.05, 'width_ratios':[3, 1.25],
                                        'bottom':0.2, 'top':0.85,
                                        'left':0.075, 'right':0.995})

    prot_text   = r'$v_\mathrm{{eq}} = {0:.1f}$ km s$^{{-1}}$, '.format(veq)
    nspots_text = r'$N_\mathrm{{spots}} = {0}$, '.format(N)
    inclin_text = r'$i_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(inc)
    obli_text   = r'$\Psi_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(obl)
    sp_deg_text = r'max Y$_{{l,m}} = {0} \:$ $\ocirc$'.format(ydeg)
    star_title  = prot_text + nspots_text + inclin_text + obli_text + sp_deg_text
    plt.suptitle(star_title, size=22)

    ax[0].set_ylabel(r"Normalized Flux", fontsize=20)
    ax[0].set_xlabel(r"Time (days)", fontsize=20)

    ax[1].tick_params(labelleft=False, labelbottom=False)
    ax[1].set_axis_off()

    ax[0].set_xlim([np.min(t), np.max(t)])
    ax[0].set_ylim([min(f)-(0.1*np.ptp(f)), max(f)+(0.1*np.ptp(f))])

    lonlines, latlines, borders = grid_lines(ax[1], theta=theta, inc=inc, obl=obl)

    ax[1].set_xlim(-1.05, 1.05)
    ax[1].set_ylim(-1.05, 1.05)
    dx = 2.0 / image.shape[1]
    extent = (-1 - dx, 1, -1 - dx, 1)

    rot_map = ax[1].imshow(image[0], cmap="plasma", extent=extent, origin="lower",
                           vmin=np.nanmin(image), vmax=np.nanmax(image), zorder=-10)

    lc, = ax[0].plot([], [], lw=1.75, color='black')

    def update(i):
        lc.set_data(t[:i], f[:i])
        rot_map.set_array(image[i])
        lines  = [lc]
        images = [rot_map]
        lons = get_ortho_longitude_lines(inc=np.deg2rad(inc), obl=np.deg2rad(obl),
                                         theta=np.deg2rad(theta[i]))
        for n, l in enumerate(lons):
            lonlines[n].set_xdata(l[0])
            lonlines[n].set_ydata(l[1])
        return tuple(lines + images + lonlines + latlines + borders)

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=75, blit=True)

    if fn is not None:
        if '.mp4' in fn:
            ani.save(fn, writer="ffmpeg")
        else:
            ani.save(fn + '.mp4', writer="ffmpeg")
    else:
        plt.show()

def plot_curve(t, y, y_err=None, plot_rv=None, plot_rv_int=False, plot_both=False,
               t_rv=None, y_rv=None, t_rv_int=None, rv_int=None, fn=None):

    if plot_both:
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 5),
                               gridspec_kw={'wspace':0.05})

        if y_err is not None:
            #ax[0].errorbar(x=t, y=y, yerr=y_err, ls='', markersize=2.0, elinewidth=1.0,
            #            barsabove=False, marker='o', alpha=0.75, mfc='black',
            #            ecolor='black', mec='black')
            ax[0].scatter(t, y, c='black', marker='o', s=10)
        else:
            ax[0].plot(t, y, color='black', lw=1.75)
        ax[0].set_ylabel(r"Normalized Flux", fontsize=20)

        ax[1].scatter(t_rv, y_rv, c='black', marker='o', s=25)
        ax[1].set_ylabel(r"RV (m s$^{-1}$)", fontsize=20)
        ax[1].set_xlabel(r"Time (days)", fontsize=20)

        if plot_rv_int:
            ax[1].scatter(t_rv_int, rv_int, edgecolors='royalblue', facecolors='white',
                          linewidths=2.0, marker='o', s=50)

        for a in ax:
            a.get_yaxis().get_major_formatter().set_useOffset(False)
            a.set_xlim([np.amin(t), np.amax(t)])

        ax[0].set_ylim([np.amin(y)-(0.1*np.ptp(y)), np.amax(y)+(0.1*np.ptp(y))])
        ax[1].set_ylim([np.amin(y_rv)-(0.1*np.ptp(y_rv)), np.amax(y_rv)+(0.1*np.ptp(y_rv))])

    else:
        fig, ax = plt.subplots(figsize=(12, 4))

        if not plot_rv:
            if y_err is not None:
                #ax.errorbar(x=t, y=y, yerr=y_err, ls='', markersize=2.0, elinewidth=1.0,
                #               barsabove=False, marker='o', alpha=0.75, mfc='black',
                #                ecolor='black', mec='black')
                ax.scatter(t, y, c='black', marker='o', s=10)
            else:
                ax.plot(t, y, color='black', lw=1.75)
            ax.set_ylabel(r"Normalized Flux", fontsize=20)

        else:
            ax.scatter(t, y, c='black', marker='o', s=10)
            ax.set_ylabel(r"RV (m s$^{-1}$)", fontsize=20)

        ax.set_xlabel(r"Time (days)", fontsize=20)

        ax.set_xlim([np.amin(t), np.amax(t)])

    if fn is not None:
        if '.png' in fn:
            plt.savefig(fn, format='png', bbox_inches='tight', dpi=400)
        elif '.pdf' in fn:
            plt.savefig(fn, format='pdf', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(fn+'.png', format='png', bbox_inches='tight', dpi=400)
            plt.savefig(fn+'.pdf', format='pdf', bbox_inches='tight', dpi=400)
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
