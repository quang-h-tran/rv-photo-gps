import numpy as np
import pandas as pd
import starry
from plot_starry import plot_animation, plot_curve
from math_utils import calc_veq, calc_prot
import scipy.stats as stats
import os
from _utils import create_dirs

starry.config.lazy = False
starry.config.quiet = True

################################################################################

def kepler_cadence(tmin=0.0, tmax=90.0):
    """
    1 Kepler quarter: 90 days with 30 min cadence
    """
    return np.arange(tmin, tmax, 30./1440.)

def tess_cadence(tmin=0.0, tmax=27.4):
    """
    1 TESS sector: 27.4 days with 2 min cadence
    """
    return np.arange(tmin, tmax, 2./1440.)

def hour_cadence(tmin=0.0, tmax=30.0):
    """
    1 month with 1 hour cadence
    """
    return np.arange(tmin, tmax, 60./1440.)

################################################################################

def decay_namekata(r0, c0, t0, t):
    """
    Namekata et al. 2019
    Decay rate of 8 MSH / hr, which we bump up 5x to 40 MSH / hr,
    equal to 24 hr /day * 40 MSH / hr to 960 MSH / day.
    Consider a typical spot size of 2x the maximum listed in Namekata+19, of
    ~5e4 MSH (5% of the hemisphere of a 1 Solar Radius star). 5% of a
    hemisphere of 180 deg is 9 deg. Let's set that as the ~max~ radius
    of a spot (more in initial_spot_params function).

    Thus, at 960 MSH / day, and a spot size of ~5e4 MSH, a spot decays at ~2%
    per day, with a total lifetime of ~50 days.
    """
    from astropy import units as u

    delta_t = np.abs(t0 - t)

    r_decay_per_day = 0.02 * r0
    c_decay_per_day = 0.04 * c0
    #r_decay_per_day = 0.004 * r0
    #c_decay_per_day = 0.008 * c0

    #print(delta_t, r_decay_per_day)

    r_new = r0 - (r_decay_per_day * delta_t)
    c_new = r0 - (r_decay_per_day * delta_t)

    return r_new, c_new

# Equation to change an individual spot parameters, # v5: 0.2, v6: 0.02, v7: 0.1
def eta(r0, c0, t0, t, tau=0.1):
    """
    Exponetial decay function for spots. Contrast decays
    twice as fast as radius.

    ::Params::
    ==========
    c0 (float or array): initial contrast of spot
    r0 (float or array): initial radius of spot
    t0 (float or array): random choice for start of spot lifetime
    t (float): current timestep in star lifetime
    tau (int, optional):
        decay factor, arbitrarily set to 0.1,
        there should be a more intelligent choice here?

    ::Returns::
    ===========
    r, c : float
        spot radius and contrast at timestep t
    """

    # calculate the time difference between the spot's maximum
    # contrast (t0) and current time (t), set to absolute
    # to account for user error and/or the set coming in vs. out
    delta_t = np.abs(t - t0)
    r, c = r0 * np.exp(-delta_t*tau), c0 * np.exp(-delta_t*tau*2.0)

    return r, c

################################################################################

def save_spot_params(r_vec, c_vec, lat_vec, lon_vec, t_vec, fn):

    spot_params = {'Radius':r_vec, 'Contrast':c_vec, 'Latitude':lat_vec,
                   'Longitude':lon_vec, 'T0':t_vec}

    spot_params = pd.DataFrame.from_dict(spot_params)

    spot_params.to_csv(fn+'.csv')

def load_spot_params(fn):

     spots = pd.read_csv(fn+'.csv')

     r_vec = spots["Radius"].to_numpy()
     c_vec = spots["Contrast"].to_numpy()
     lat_vec = spots["Latitude"].to_numpy()
     lon_vec = spots["Longitude"].to_numpy()
     t_vec = spots["T0"].to_numpy()

     return r_vec, c_vec, lat_vec, lon_vec, t_vec

#def initial_spot_params(N, t, rmax=10.0, rmin=2.0, cmax=0.3, cmin=0.1):
def initial_spot_params(N, t, tau, rmax=10.0, rmin=1.0, cmax=0.3, cmin=0.05): # N=50
#def initial_spot_params(N, t, rmax=7.5, rmin=1.5, cmax=0.225, cmin=0.075): # N=25, t=20
    """
    Randomly generate arrays of parameters to create spots parameters drawn
    from a uniform prior distribution.

    ::Params::
    ==========
    N (int): number of spots
    t (array): time array of the light curve
    rmax (float, optional): maximum spot radius, default to 15
    rmin (float, optional): minimum spot radius, default to 3.0 deg
    cmax, cmin (floats, optional):
        max/min of contrast, defaul to 0.1 and 0.25.

    ::Returns::
    ===========
    r_vec, c_vec, lat_vec, lon_vec, t_vec : arrays
        arrays of len N. lat ranges from -90 to 90 and l -180 to 180 [deg]
        t ranges from 0 to maximum [arbitrary units]
    """
    r_vec = np.random.uniform(low=rmin, high=rmax, size=N)
    c_vec = np.random.uniform(low=cmin, high=cmax, size=N)
    lat_vec = np.random.uniform(low=-90., high=90., size=N)
    #lat_vec = np.random.uniform(low=[-90., 30.], high=[-30., 90.], size=(int(N/2)+1, 2))
    #lat_vec = lat_vec.flatten()[:N]
    lon_vec = np.random.uniform(low=-180, high=180, size=N)
    t_vec = np.random.uniform(low=min(t)-(1./tau), high=max(t)+(1./tau), size=N)

    return r_vec, c_vec, lat_vec, lon_vec, t_vec

def angle_diff(lat, lon, spot_lat, spot_lon):
    """
    Calculate the angular difference between spot center and the rest of the (spherical) stellar surface.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    spot_lat, spot_lon = np.deg2rad(spot_lat), np.deg2rad(spot_lon)

    delta_lon = np.abs(lon - spot_lon)
    sines = np.sin(lat) * np.sin(spot_lat)
    cosines = np.cos(lat)*np.cos(spot_lat)*np.cos(delta_lon)

    return np.rad2deg(np.arccos(sines+cosines))

def spot_c(arc_length, spot_radius):
    """
    Treat the spot contrast radially from spot center, normalized by max contrast, as a Gaussian.
    """
    norm = stats.norm.pdf(0.0, 0.0, spot_radius)
    return stats.norm.pdf(arc_length, 0.0, spot_radius) / norm

def add_spot(p, spot_lats, spot_lons, spot_rads, spot_cons, lat, lon, N_spot=1):
    """
    Add a spot on the stellar surface map at the pixel level
    """
    p_i = p.copy()

    for s in range(N_spot):
        arc_length = angle_diff(lat, lon, spot_lats[s], spot_lons[s])
        p_i *= (1.0 - spot_cons[s]*spot_c(arc_length, spot_rads[s]))

    return p_i

def spotted_map(N, inc, obl, ydeg=20, udeg=2, q1=0.2, q2=0.1, alpha=0.0,
           t=None, t_step=None, t_min=0.0, t_max=20.0,
           veq=5., prot=None, rstar=1.0, animate=False,
           calc_rv=True, rv_cadence=None, rv_int=None,
           f_err=False, rv_err=False, spot_params_fn=None,
           save_spots=True, load_spots=False,
           spot_param_vec=None, tau=0.1):
    """
    Calculate the light curve flux and radial velocity curve of a star (i.e.,
    a starry map) with a population of spots.

    Change Log:
    v. 28.09.21
        No occultor. Removed all calls to starry.map that were unnecessary,
        especially in the for loop to add spots, as well as in the add_spot
        function itself. Reduces computation time by ~2 orders of magnitude.
    v. 02.08.21
        No occultor. RVs. Spots added via "get_pixel_transforms" instead of
        "spots". Note: Add rv_cadence, with rv_int parameter that averages
        across all points within RV_midpoint +/- 0.5*RV_int. Can do this with
        a linearly interpolated version of the RV curve, since it cost very
        little to also compute the supersampled (i.e., sampled at the
        photometric light curve cadence)
    v. 23.08.21
        No occultor. Light curve and map.

    ::Args::
    ==========
    N (int): Number of spots.
    inc (float): Inclination of star in deg.
    obl (float): Stellar obliquity in deg.
    ydeg (int, optional): Max spherical harmonic angle, defaults to 20.
    udeg (int, optional): Limb-darkening degree. Defaults to 2.
    q1 (float, optional): First limb-darkening coefficient. Defaults to 0.2.
    q2 (float, optional): Second limb-darkening coefficient.
        Only used if udeg == 2. Defaults to 0.1.
    t (array-like or int, optional): Time over which to compute
        the flux/rv curve. If array, uses that as timestep.
        If int, then calculates time array as np.linspace(t_min, t_max, t).
        t_step argument overrides this one. Units of [days].
        Defaults to np.linspace(t_min, t_max, 50).
    t_step (float, optional): Time step. If this is given, will supercede
        t parameter. Constructs array as np.arange(t_min, t_max, t_step).
    t_min (float, optional): Minimum time. Defaults to 0 [days].
    t_max (float, optional): Maximum time. Defaults to 30.0 [days].
    veq (float, optional): Equatorial velocity of the star. Defaults to 5 km/s.
    prot (float, optional): Stellar rotational period in [days].
        This is used to calculate the rotation speed. Defaults to None but
        then calculated using veq and rstar. If parameter is given,
        will supercede veq parameter and calculate based on prot and rstar.
    rstar (float, optional): stellar radius in Rsun, used to calculate eq vel
        or rot per. Defaults to 1.0 solar radius.
    animate (boolean, optional): Return image array for animation. Defaults to False.
    calc_rv (boolean, optional): Defaults to True. Returns rv array.
    rv_cadence (array, optional): If not None, calculates RVs at the time
        position specified. The returned rv array is calculated as the average
        of all RV points at the rv_cadence position +/- 0.5*rv_int. The RV
        points average are taken from a supersampled, linearly interpolated
        version of the RV curve.
    rv_int (array, optional): If not None, used to calculate RV values at
        rv_cadence values. If rv_cadence is given but rv_int is not, rv_int
        defaults to 300 s (5 min). Units of seconds.

    ::Returns::
    ===========
    t (array): Time array.
    theta (array): Theta array of length t.
    f (array): Flux array of length t.

    rv (array): RV array. Length t or length rv_cadence.
    t_rv (array): RV time array.
    """

    if t_step is not None:
        t = np.arange(t_min, t_max, t_step)
    elif t is not None:
        if isinstance(t, np.ndarray):
            t = t
        elif isinstance(t, int):
            t = np.linspace(t_min, t_max, t)
    else:
        t = np.linspace(t_min, t_max, 50)

    if animate:
        res = 300
        image = np.empty((len(theta), res, res))

    if prot is None:
        prot = calc_prot(veq, rstar=rstar)
    else:
        veq = calc_veq(prot, rstar=rstar)

    theta = 360.0 / prot * t

    # Create the stellar map using input orientation
    map = starry.Map(ydeg=int(ydeg), inc=inc, obl=obl,
                     udeg=int(udeg), rv=calc_rv)
    if udeg >= 1:
        map[1] = q1
    if udeg == 2:
        map[2] = q2
    if calc_rv:
        map.veq = veq * 1e4
        map.alpha = alpha

        # Calculate the B matrix (for the RVs)
        rv_f = map.ops.compute_rv_filter(map._inc, map._obl, map._veq, map._alpha)
        kwargs = {"xo":0.0, "yo":0.0, "zo":1.0, "ro":0.0, "theta":theta}
        theta_hold, xo, yo, zo, ro = map._get_flux_kwargs(kwargs)
        B = map.ops.X(theta_hold, xo, yo, zo, ro, map._inc, map._obl, map._u, rv_f)

    A = map.design_matrix(theta=theta) # design matrix, A
    y_nt = np.zeros((len(map.y), len(t))) # y matrix of y vectors at time t
    lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms() # pixel transformations
    p = Y2P.dot(map.y) # pixels

    # Create (or load) spots
    #if spot_params_fn is None:
    #    spot_params_base = 'Simulated Data/' + '/N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}/'.format(N, inc, obl, prot)
    #    spot_params_end = 'spot_params'
    #    spot_params_fn = spot_params_base + spot_params_end

    if spot_param_vec is None:
        if load_spots:
            r_vec, c_vec, lat_vec, lon_vec, t_vec = load_spot_params(spot_params_fn)
            #t_vec *= 0.0
        else:
            r_vec, c_vec, lat_vec, lon_vec, t_vec = initial_spot_params(N, t, tau)
    else:
        r_vec, c_vec, lat_vec, lon_vec, t_vec = spot_param_vec
    if save_spots:
        save_spot_params(r_vec, c_vec, lat_vec, lon_vec, t_vec, spot_params_fn)

    for i in range(len(t)): # At every time step, calculate the flux

        r, c = eta(r_vec, c_vec, t_vec, t[i], tau=tau)  # Decay spots
        p_i = add_spot(p, lat_vec, lon_vec, r, c, N_spot=N,
                       lat=lat, lon=lon) # pixel-level surface map changes

        if animate:
            image[i] = map.render(theta=theta[i], res=res) # map image at t

        y_nt[:,i] = P2Y.dot(p_i)

    # Take the full design matrix and dot it into spherical harmonic matrix
    # Need to take the diagonal since you want the flux at each time step, i
    f = np.diagonal(A.dot(y_nt)).copy()
    if calc_rv:
        rv = np.diagonal(B.dot(y_nt)).copy() / f

    # Add photometric error to light curve
    if f_err:
        if isinstance(f_err, np.ndarray):
            f_err = f_err
        elif isinstance(f_err, float):
            f_err = f_err * np.random.randn(len(f))
        else:
            f_err = 2.5e-4 * np.random.randn(len(f)) # Kepler?

        f += f_err
    else:
        f_err = None

    # Add instrumental error to RV curve
    if rv_err:
        if isinstance(rv_err, np.ndarray):
            rv_err = rv_err
        elif isinstance(rv_err, float):
            rv_err = rv_err * np.random.randn(len(rv))
        else:
            #rv_err = 15. * np.random.randn(len(rv))
            rv_err = 20. * np.random.randn(len(rv))

        rv += rv_err
    else:
        rv_err = None

    if rv_cadence is not None:
        from scipy.interpolate import interp1d

        sec_to_day = 1.157e-5

        # default to 300 second integration time (HPF)
        if rv_int is None:
            rv_int = 300.

        rv_int_range = 0.5*rv_int*sec_to_day

        # interpolate rv curve
        rv_interp_func = interp1d(t, rv, kind='cubic')

        sampled_rv = np.zeros_like(rv_cadence)
        for i in range(len(rv_cadence)):
            # average all rv values at rv_cadence -0.5*rv_int to +0.5*rv_int
            rv_cadence_int = np.arange(rv_cadence[i] - rv_int_range, rv_cadence[i] + rv_int_range, 1.157e-5)
            sampled_rv[i] = np.nanmean(rv_interp_func(rv_cadence_int))

    returns = [t, f]

    if f_err is not None:
        returns += [f_err]
    if animate:
        returns += [theta, image]
    if calc_rv:
        returns += [rv]
        if rv_err is not None:
            returns += [rv_err]
        if rv_cadence is not None:
            returns += [sampled_rv, rv_cadence]

    return returns

################################################################################
################################################################################
################################################################################

#N, inc, obl, prot, veq = 25, 85.0, 5.5, 7.2, calc_veq(7.2)
#N, inc, obl, prot, veq = 5, 78.0, 4.1, 4.8, calc_veq(4.8)
#N, inc, obl, prot, veq = 15, 87.2, 4.5, 5.8, calc_veq(5.8)
#N, inc, obl, prot, veq = 37, 43.2, 3.2, 12.2, calc_veq(12.2)
#N, inc, obl, prot, veq = 12, 80.4, 6.2, 3.1, calc_veq(3.1)
# just like above but with spots pushed to poles
#N, inc, obl, prot, veq = 12, 80.3, 6.3, 3.1, calc_veq(3.1)
#N, inc, obl, prot, veq = 21, 72.1, 9.1, calc_prot(22.5), 22.5
# just like above but with spots pushed to poles
#N, inc, obl, prot, veq = 21, 72.0, 9.2, calc_prot(22.5), 22.5
# just like above but spots are bigger,
# the other two have smaller spots than normal
#N, inc, obl, prot, veq = 21, 71.9, 9.3, calc_prot(22.5), 22.5

#N, inc, obl, prot, veq = 25, 88.0, 4.0, 8.0, calc_veq(8.0)
#N, inc, obl, prot, veq = 50, 86.5, 6.0, 11.0, calc_veq(11.0)
N, inc, obl, prot, veq = 45, 85.3, 4.0, 9.0, calc_veq(9.0)

#t = hour_cadence(tmin=0.0, tmax=27.4)

N_ori = 1
taus = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
#taus = [0.005, 0.0033, 0.0025, 0.002]

#taus = [0.1]

def vary_tau(taus, t=kepler_cadence(tmin=0.0)):

    paths = create_dirs(N_ori, N, inc, obl, prot, veq)
    sim_data_path = paths[0]
    decay_rate_dir = sim_data_path + "Decay Rate/"
    if not os.path.isdir(decay_rate_dir):
        os.mkdir(decay_rate_dir)

    for tau in taus:

        decay_rate_path = decay_rate_dir + "/tau_{0:.3f}".format(tau)

        t, f, f_err, rv, rv_err = spotted_map(N=N, inc=inc, obl=obl, prot=prot,
                                  t=t, calc_rv=True, f_err=True, rv_err=True, tau=tau,
                                  load_spots=True, spot_params_fn=sim_data_path+"spot_params")

        #f_prime = np.gradient(f - f_err, t)
        #f_prime_err = 2.5e-4 * np.random.randn(len(f_prime))

        #np.save(decay_rate_path + "_f_prime.npy", f_prime)
        #np.save(decay_rate_path + "_f_prime_err.npy", f_prime_err)
        np.save(decay_rate_path + "_t.npy", t)
        np.save(decay_rate_path + "_f.npy", f)
        np.save(decay_rate_path + "_ferr.npy", f_err)
        np.save(decay_rate_path + "_rv.npy", rv)
        np.save(decay_rate_path + "_rverr.npy", rv_err)
        np.save(decay_rate_path + "_f_no_err.npy", f - f_err)

        #plot_curve(t, f, y_err=f_err, t_rv=t, y_rv=rv, plot_rv=True, plot_both=True)
        #plot_curve(t, f - f_err, plot_rv=False, plot_both=False)

vary_tau(taus)

vars = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 2.0]

def vary_spot_var(vars, rad=True, con=False, tau=0.1, t=kepler_cadence(tmin=0.0)):

    paths = create_dirs(N_ori, N, inc, obl, prot, veq)
    sim_data_path = paths[0]

    if rad and con:
        var_dir = sim_data_path + "Spot Radius and Contrast/"
        var_name = ""
    elif rad and not con:
        var_dir = sim_data_path + "Spot Radius/"
        var_name = "rad_"
    elif not rad and con:
        var_dir = sim_data_path + "Spot Contrast/"
        var_name = "con_"

    if not os.path.isdir(var_dir):
        os.mkdir(var_dir)

    for var in vars:

        var_path = var_dir + "/"+var_name+"{0:.3f}".format(var)

        r_vec, c_vec, lat_vec, lon_vec, t_vec = load_spot_params(sim_data_path+"spot_params")

        if rad and con:
            new_spot_params = (r_vec*var, c_vec*var, lat_vec, lon_vec, t_vec)
        elif rad and not con:
            new_spot_params = (r_vec*var, c_vec, lat_vec, lon_vec, t_vec)
        elif not rad and con:
            new_spot_params = (r_vec, c_vec*var, lat_vec, lon_vec, t_vec)

        base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)

        t, f, f_err, rv, rv_err = spotted_map(N=N, inc=inc, obl=obl, prot=prot,
                                  t=t, calc_rv=True, f_err=2.5e-4*var, rv_err=20.*var, tau=tau,
                                  load_spots=True, save_spots=True, spot_params_fn=sim_data_path+"spot_params",
                                  spot_param_vec=new_spot_params)

        f_prime = np.gradient(f - f_err, t)
        f_prime_err = 2.5e-4 * np.random.randn(len(f_prime))

        np.save(var_path + "_f_prime.npy", f_prime)
        np.save(var_path + "_f_prime_err.npy", f_prime_err)
        np.save(var_path + "_t.npy", t)
        np.save(var_path + "_f.npy", f)
        np.save(var_path + "_ferr.npy", f_err)
        np.save(var_path + "_rv.npy", rv)
        np.save(var_path + "_rverr.npy", rv_err)

        #plot_curve(t, f, y_err=f_err, t_rv=t, y_rv=rv, plot_rv=True, plot_both=True)

################################################################################
######################## Example Code to Test Functions ########################
################################################################################

# Create animation
"""
t, f, f_err, theta, image = spotted_map(N=N, inc=inc, obl=obl, prot=prot, udeg=1,
                          #t=100, t_min=0.0, t_max=20.0,
                          t=t, animate=True, f_err=True, calc_rv=False)

np.save("Simulated Data/variable_test_theta" + base_fn, theta)
np.save("Simulated Data/variable_test_image" + base_fn, image)

ani_fn = 'Plots/variable_test_animation_N{0:d}_i{1:.1f}_P{2:.1f}_t{3:d}.mp4'.format(N, inc, prot, len(t))
plot_animation(t, f, theta, image, N=N, inc=inc, obl=obl, ydeg=20.0, veq=veq, fn=ani_fn)
"""

### RV cadence test
"""rv_cadence = np.array([2.15, 3.2, 4.10, 4.135, 6.12, 9.19, 10.16,
                       13.09, 14.11, 16.2, 17.19, 18.18])
rv_int = 1800.

t, f, rv, samp_rv, t_rv = spotted_map(N=N, inc=inc, obl=obl, prot=prot, udeg=1,
                       t=200, t_min=0.0, t_max=20.0, calc_rv=True,
                       rv_cadence=rv_cadence, rv_int=rv_int)

plot_curve(t, f, t_rv=t, y_rv=rv, rv_int=samp_rv, t_rv_int=t_rv,
           plot_rv=True, plot_both=True, plot_rv_int=True,
           fn='Plots/rv_cadence_test_v2.png')"""


# Plotting
"""
plot_fn = "example_file"

# Plot Example #1:
# Plot both photometric and RV curve, no RV cadence.
plot_curve(t, f, t_rv=t, y_rv=rv, plot_rv=True,
           plot_both=True, fn=plot_fn)

# Plot Example #2:
# Plot only RV curve, no RV cadence
plot_curve(t, rv, plot_rv=True, fn=plot_fn)

# Plot Example #3
# Plot only LC.
plot_curve(t, f, fn=plot_fn)

# Plot Example #4
# Plot animation of light curve and spot map.
ani_fn = "example_image"
plot_animation(t, f, theta, image, N, inc, obl, ydeg, veq, fn=ani_fn)
"""



# Example taken from starry readthedocs.io
"""lat_vec = np.array([-60.0, -30.0, 30.0, 60.0, -60.0, -30.0, 30.0, 60.0])
lon_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

r_vec = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) # degrees
c_vec = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

map = starry.Map(ydeg=20, udeg=1, rv=True)
map.inc = incs[2]
map.obl = obls[1]
prot = prots[1]
map.veq = calc_veq(prot) * 1e4
map.alpha = 0.0
map[1] = 0.2
#map[2] = 0.1

t = np.linspace(0.0, 30.0, 20)
theta = 360.0 / prot * t

#map.spot(contrast=0.5, radius=20.0, lat=30, lon=30)

add_spot(map, lat_vec, lon_vec, r_vec, c_vec, N_spot=8)"""
