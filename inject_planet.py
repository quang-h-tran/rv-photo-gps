from astropy import units as u
from astropy import constants as const
import numpy as np

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

################################################################################

def Newton_Raphson_factor(E0, M, e):
    top = M - E0 + (e*np.sin(E0))
    bot = 1. - (e * np.cos(E0))

    return top/bot

def Newton_Raphson(M, e, E0):
    correct = Newton_Raphson_factor(E0, M, e)
    return correct

def inter_NR(M, e, E0, n):
    E = E0
    for i in range(n):
        E_n = Newton_Raphson_factor(E, M, e)
        E = E_n + E

    return E

def x(a, E, e):
    return a * (np.cos(E) - e)

def y(a, e, E):
    return a * np.sqrt(1 - e**2.) * np.sin(E)

def f(e, E):
    tan_f_2 = np.sqrt((1 + e)/(1 - e)) * np.tan(E/2.)
    f_2 = np.arctan(tan_f_2)
    return f_2 * 2.

def T(a):
    return a**(3./2.)

def M(t, T, tau=0):
    return ((2.*np.pi) / T) * (t - tau)

################################################################################

G = const.G
G = G.to(u.AU**3 / (u.Msun * u.yr**2))

"""
What you need to define in order to inject the planet:

1a) semi-major axis (AU) OR
1b) orbital period (year)
2) eccenticity - set to 0.0 by default
3) inclination (deg) - taken from starry configuration
4) m_star (solar mass) - set to 1.0, as in starry
5) m_planet (jupiter mass) - set to 1.0 by default
6) tau0 (year) - set to 0.0 by default
"""

def period(a, m2, m1=1.0*u.Msun):
    a *= u.AU
    m2 *= u.Mjup

    top = 4. * np.pi**2. * a**3.
    bot = G * (m1 + m2)

    return np.sqrt(top/bot).to(u.yr).value

def semi_major_axis(P, m2, m1=1.0*u.Msun):
    P *= u.year
    m2 *= u.Mjup

    top = P**2 * G * (m1 + m2)
    bot = 4. * np.pi**2.

    return ((top / bot)**(1/3)).to(u.AU).value

def K(m2, i, e, m1=1.0*u.Msun, *, T=None, a=None):

    m2 = (m2 * u.Mjup).to(u.Msun)

    if T is None and a:
        a_hold = (((m1)/(m1+m2))*a).value
        T = period(a_hold, m2.value, m1=m1)
    else:
        a_hold = semi_major_axis(T, m2.value) * u.AU
        a_hold = (((m1)/(m1+m2))*a_hold).value
        T = period(a_hold, m2.value, m1=m1)

    T *= u.yr

    top = (2.*np.pi*G)**(1/3) * m2 * np.sin(i)
    bot = T**(1/3) * np.sqrt(1. - e**2.) * m1**(2/3)
    # AU / yr
    return (top/bot).to(u.m / u.s)

def V(semi_amp, w, true_anom, e=0.0, V0=0.):
    return V0 + semi_amp*(np.cos(w + true_anom) + e*np.cos(w))

# I want a HJ: P = 10 days, M = 1.5 MJup
# I want a warm saturn: P = 50 days, M = 0.3 MJup
HJ_P = (9.0 * u.day).to(u.year).value
HJ_m = 2.0 # Jupiter units
HJ_i = np.deg2rad(86.5) # edge on
HJ_e = 0.0
HJ_t0 = 0.0
HJ_w = 0.0

K_HJ = K(HJ_m, HJ_i, HJ_e, T=HJ_P)
print(K_HJ)
t_range = np.linspace(HJ_t0, HJ_t0+(HJ_P*10.), 1000)

M0 = M(t_range, HJ_P, tau=15.0)
M1 = M(t_range, (20.25 * u.day).to(u.year).value, tau=15.0)

E = inter_NR(e=HJ_e, M=M0, E0=1.0, n=20)
E1 = inter_NR(e=HJ_e, M=M1, E0=1.0, n=20)

true_anom = f(HJ_e, E)
true_anom1 = f(HJ_e, E1)

HJ_V = V(K_HJ, HJ_w, true_anom, HJ_e) # m/s
HJ_V1 = V(203.59*(u.m / u.s), HJ_w, true_anom1, HJ_e) # m/s

fig, ax = plt.subplots(figsize=(15,7))
ax.plot(t_range*365., HJ_V, label=r'Planet', color='blue', linewidth=2.5)
ax.plot(t_range*365., HJ_V1, label=r'Planet 2', color='orange', linewidth=2.5)

#ax.set_xlabel(r"Period (yr)", size=24)
ax.set_xlabel(r"Time (day)", size=24)
ax.set_ylabel(r"RV (m s$^{-1}$)", size=24)

plt.legend(framealpha=1.0, facecolor='white', edgecolor='white', fontsize=36)
plt.tight_layout()
plt.show()
