import numpy as np
import sympy as sm
import matplotlib.pyplot as plt

A, l, gamma, P, x_ = sm.symbols("A, l, gamma, P, x", real=True, positive=True)
t_1, t_2 = sm.symbols("t_1, t_2", real=True)

l_inverse_square = 1 / (2 * l**2)
sq_exp = -(l_inverse_square * (t_1 - t_2)**2)
sin_sq = -gamma**2 * sm.sin((sm.pi*(t_1 - t_2)) / P)**2

k   = A * sm.exp(sq_exp + sin_sq)
kp  = sm.simplify( sm.diff(k, t_2).expand() )
kpp = sm.simplify( sm.diff(sm.diff(k, t_1), t_2).expand() )

print("")
print(k)
print("")
print(kp)
print("")
print(kpp)
print("")
print("for QuasiPer:", sm.simplify(sm.diff(kpp, x_).subs([(x_, 0)])))
print("")

def quasi_per(t1, t2, A, l, gamma, P):
    return A*np.exp(-gamma**2*np.sin(np.pi*(t1 - t2)/P)**2 - (t1 - t2)**2/(2*l**2))

def quasi_per_cross(t1, t2, A, l, gamma, P):
    top = A*(P*t1 - P*t2 + np.pi*gamma**2*l**2*np.sin(2*np.pi*(t1 - t2)/P))
    bot = (P*l**2)
    return top*np.exp(-gamma**2*np.sin(np.pi*(t1 - t2)/P)**2 - t1**2/(2*l**2) + t1*t2/l**2 - t2**2/(2*l**2))/bot

def quasi_per_grad(t1, t2, A, l, gamma, P):
    top = A*(P**2*l**2 - P**2*t1**2 + 2*P**2*t1*t2 - P**2*t2**2 - 2*np.pi*P*gamma**2*l**2*t1*np.sin(2*np.pi*(t1 - t2)/P) + 2*np.pi*P*gamma**2*l**2*t2*np.sin(2*np.pi*(t1 - t2)/P) - 4*np.pi**2*gamma**4*l**4*np.sin(np.pi*(t1 - t2)/P)**2*np.cos(np.pi*(t1 - t2)/P)**2 - 2*np.pi**2*gamma**2*l**4*np.sin(np.pi*(t1 - t2)/P)**2 + 2*np.pi**2*gamma**2*l**4*np.cos(np.pi*(t1 - t2)/P)**2)
    bot = (P**2*l**4)
    return top*np.exp(-gamma**2*np.sin(np.pi*(t1 - t2)/P)**2 - t1**2/(2*l**2) + t1*t2/l**2 - t2**2/(2*l**2))/bot

"""plt.figure()
t = np.linspace(-5.0, 5.0, 10000)
plt.plot(t, quasi_per(t, 0, 1, 1, 1, 1), label="k(t1, t2)")
plt.plot(t, quasi_per_cross(t, 0, 1, 1, 1, 1), label="dk(t1, t2)/dt2")
plt.plot(t, quasi_per_grad(t, 0, 1, 1, 1, 1), label="$d^2k(t1, t2)/(dt1 dt2)$")
plt.legend()
plt.title("Quasi-Per.")
plt.xlabel("t1 - t2")
plt.show()"""

def sample_gp(random, K, size=None):
    return random.multivariate_normal(np.zeros(K.shape[0]), K, size=size)

t = np.linspace(0, 10, 1501)
tp = np.linspace(0, 10, 1800)

K = quasi_per(t[:, None], t[None, :], 1, 1, 1, 1)
Kp = quasi_per_cross(t[:, None], tp[None, :], 1, 1, 1, 1)
KpT = quasi_per_cross(tp[None, :], t[:, None], 1, 1, 1, 1)
Kpp = quasi_per_grad(tp[:, None], tp[None, :], 1, 1, 1, 1)
cov = np.concatenate((
    np.concatenate((K, Kp), axis=1),
    np.concatenate((Kp.T, Kpp), axis=1),
), axis=0)
y = sample_gp(np.random.default_rng(1000), cov)

"""plt.title("SqExp: covariance matrix")
plt.imshow(cov)
plt.xticks([])
plt.yticks([])"""


plt.figure()
plt.plot(t, y[:len(t)], label="f(t)")
plt.plot(tp, y[len(t):], label="f'(t)")
plt.plot(t, 1/(t[1]-t[0])*np.gradient(y[:len(t)]))
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("t")
plt.ylabel("f(t); f'(t)")
plt.legend()
plt.xlim(t.min(), t.max())
plt.title("Quasi-Per");
plt.show()
