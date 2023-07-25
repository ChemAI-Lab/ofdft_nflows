import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.integrate import simps
import matplotlib.pyplot as plt
import copy

N = 256
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]


def hartree_poisson(rho):
    v_h = np.zeros(N) + 1e-16
    v_h_n = np.zeros(N)
    etol = 1e-5
    diff = np.ones(N)
    dt = dx ** 2 * 0.1
    while np.min(diff) > etol:
        v_h_n[0] = v_h[0] + (dt / (dx**2)) * (v_h[-1] +
                                              v_h[1] - 2 * v_h[0]) + rho[0] * dt
        v_h_n[-1] = v_h[-1] + (dt / (dx**2)) * \
            (v_h[-2] + v_h[0] - 2 * v_h[-1]) + rho[-1] * dt
        for i in range(1, N-1):
            v_h_n[i] = v_h[i] + (dt / (dx**2)) * \
                (v_h[i-1] + v_h[i+1] - 2 * v_h[i]) + rho[i] * dt
        diff = abs(v_h_n - v_h)
        v_h = copy.copy(v_h_n)
    return v_h


def hartree_real(rho, alpha=1e-6):
    vh = []
    for idx, xi in enumerate(x):
        # i0 = np.where(x != xi)[0]
        z = np.sqrt((x-xi)**2 + alpha**2)
        int_ = rho / z
        integ_ = simps(int_, dx=dx)
        vh.append(integ_)
    return np.array(vh) * alpha / N


def hartree_fft(rho):
    dk = 2 * np.pi / dx
    k = (np.arange(N) - N // 2) * dk
    k[N//2] = 1
    invk = 1 / k
    invk[N//2] = 0.
    rho_g = fft(rho)
    v_h = rho_g * (invk**2)
    # v_h[N//2] = 0.
    return ifft(v_h).real * N**2


def norm(inp):
    return inp / np.sum(inp)


def load_true_results(n_particles: int):
    import numpy as onp
    d_ = f'Data_1D_GaussMixPot/true_rho_grid_Ne_{n_particles}.txt'
    data = np.array(onp.loadtxt(d_))
    return data


# rho = np.loadtxt("true_rho.dat")
data = load_true_results(1)
x = data[:, 0]
rho = data[:, 1]
hp = hartree_poisson(rho)
hf = hartree_fft(rho)
hr = hartree_real(rho)
print((hp * rho).sum() * dx)
print((hf * rho).sum() * dx)
print((hr * rho).sum() * dx)
plt.plot(x, hp, '-.', label='poisson')
plt.plot(x, hr, '--', label='real')
# plt.plot(x, hf, label='fft')
plt.legend()
plt.tight_layout()
plt.show()
