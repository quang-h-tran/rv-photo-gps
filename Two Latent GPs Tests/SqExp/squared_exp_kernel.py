import numpy as np
import sympy as sm
import matplotlib.pyplot as plt

sigma, l, x_ = sm.symbols("sigma, l, x", real=True, positive=True)
t_1, t_2 = sm.symbols("t_1, t_2", real=True)

l_inverse_square = 1 / (2 * l**2)
sq_exp_ker = sigma * sm.exp(- l_inverse_square * sm.Abs(t_1 - t_2)**2)

sq_exp_ker_p = sm.simplify(sm.diff(sq_exp_ker, t_2).expand())
sq_exp_ker_pp = sm.simplify(sm.diff(sm.diff(sq_exp_ker, t_1), t_2).expand())

print("for SqExp:", sm.simplify(sm.diff(sq_exp_ker_pp, x_).subs([(x_, 0)])))

def sq_exp(t1, t2, *, sigma=1.0, l=1.0):
    x = (1 / (2 * l**2)) * np.abs(t1 - t2)**2
    return sigma * np.exp(-x)

def sq_exp_cross(t1, t2, *, sigma=1.0, l=1.0):
    return sigma*(t1 - t2)*np.exp((-t1**2/2 + t1*t2 - t2**2/2)/l**2)/l**2

def sq_exp_grad(t1, t2, *, sigma=1.0, l=1.0):
    return sigma*(l**2 - t1**2 + 2*t1*t2 - t2**2)*np.exp((-t1**2/2 + t1*t2 - t2**2/2)/l**2)/l**4

"""plt.figure()
t = np.linspace(-5.0, 5.0, 200)
plt.plot(t, sq_exp(t, 0), label="k(t1, t2)")
plt.plot(t, sq_exp_cross(t, 0), label="dk(t1, t2)/dt2")
plt.plot(t, sq_exp_grad(t, 0), label="$d^2k(t1, t2)/(dt1 dt2)$")
plt.legend()
plt.title("Squared Exp.")
plt.xlabel("t1 - t2")
plt.show()"""

def sample_gp(random, K, size=None):
    return random.multivariate_normal(np.zeros(K.shape[0]), K, size=size)

t = np.linspace(0, 10, 151)
tp = np.linspace(0, 10, 180)

K = sq_exp(t[:, None], t[None, :])
Kp = sq_exp_cross(t[:, None], tp[None, :])
KpT = sq_exp_cross(tp[None, :], t[:, None])
Kpp = sq_exp_grad(tp[:, None], tp[None, :])
cov = np.concatenate((
    np.concatenate((K, Kp), axis=1),
    np.concatenate((Kp.T, Kpp), axis=1),
), axis=0)
y = sample_gp(np.random.default_rng(1000), cov)

"""plt.title("SqExp: covariance matrix")
plt.imshow(cov)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.plot(t, y[:len(t)], label="f(t)")
plt.plot(tp, y[len(t):], label="f'(t)")
plt.plot(t, 1/(t[1]-t[0])*np.gradient(y[:len(t)]))
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("t")
plt.ylabel("f(t); f'(t)")
plt.legend()
plt.xlim(t.min(), t.max())
plt.title("SqExp");
plt.show()
"""
