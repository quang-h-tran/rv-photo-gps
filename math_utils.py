import numpy as np

def calc_veq(P, rstar=1.0):
    """
    Calculate the equatorial velocity (in km/s)
    from the rotation period (in days) and stellar radius (in Rsolar)
    """
    from astropy import units as u

    P *= u.day
    rstar *= u.Rsun

    return (2. * np.pi * rstar / P).to(u.km / u.s).value

def calc_prot(veq, rstar=1.0):
    """
    Calculate rotational period in days given equatorial velocity in km/s and
    stellar radius in Rsun.
    """
    from astropy import units as u

    veq *= (u.km/u.s)
    rstar *= u.Rsun

    return (2. * np.pi * rstar / veq).to(u.day).value
