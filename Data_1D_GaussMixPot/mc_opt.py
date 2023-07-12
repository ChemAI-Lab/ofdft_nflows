from scipy.stats import norm
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
nodes = 256  # the last node does not count for integration but is included for simpler symmetrization
L = 10
x = np.linspace(-L/2, L/2, nodes)
k = 1.0
dx = x[1] - x[0]
v = - 1.0 * np.exp(-(x-0.5)**2) - 2.0 * np.exp(-(x + 1)**2)


def chemicalPotential(rho):
    return ((deltaTdeltaRho(rho) + v + hartree(rho))).sum() * dx


def exactSolution(rho):
    mu = chemicalPotential(rho)
    solution = mu - v
    solution *= 2 / np.pi**2
    print(mu, solution)
    return np.nan_to_num(solution**0.5)


def deltaTdeltaRho(rho):
    return np.pi**2 * rho ** 2 / 4


def tf(rho):
    return np.pi**2 * rho**3 / 12


def gradient(rho):
    deriv = np.zeros_like(rho)
    for i in range(rho.shape[0]):
        prev = rho[i - 1]
        next = rho[(i + 1) % rho.shape[0]]
        deriv[i] = (next-prev) / dx / 2.0
    return deriv


def vw(rho):
    return gradient(rho)**2 / rho / 8.


def kinetic(rho, alpha=1.0, beta=0.0):
    return alpha * tf(rho)


def funcDeriv(rho):
    return deltaTdeltaRho(rho) + v + hartree(rho)


def external(rho):
    return v


def exchangePotential(rho):
    A = - (3. / np.pi)**(1. / 3.)
    return A * rho**(1. / 3.)


def exchangeEnergy(rho):
    A = - (3. / 4.) * (3. / np.pi)**(1. / 3.)
    return A * rho**(4. / 3.)


def correlationPotential(rho):
    a = -np.log(np.sqrt(2. * np.pi)) - 0.75
    b = 0.359933
    return 4 * a * rho + 2**(3/2.) * 5 * b * rho ** (3/2.) / 2


def correlationEnergy(rho):
    a = -np.log(np.sqrt(2. * np.pi)) - 0.75
    b = 0.359933
    return 2 * a * rho**2 + 2**(3/2.) * b * rho**(5/2.)


def energy(rho):
    edens = kinetic(rho) + external(rho) * rho + 0.5 * hartree(rho) * rho
    return edens.sum()*dx


def hartree(rho):
    dk = 2 * np.pi / dx
    k = (np.arange(nodes) - nodes // 2) * dk
    k[nodes//2] = 1
    invk = 1 / k
    invk[nodes//2] = 0.
    rho_g = fft(rho)
    v_h = rho_g * (invk**2)
    rho_g = fft(rho)
    v_h = rho_g / k**2
    v_h[nodes//2] = 0.
    return ifft(v_h).real

# SHO
# v = 0.5 * k * x**2

# soft coloumb
# v = -1 / (x**2 + 0.01**2)**0.5


# rho is normalized?
n_elec = 2
rho = n_elec * np.ones_like(x) / L
# charge changes during optimization and will be corrected to 1, charge from the last node flows into the middle, however last node charge is not counted in integrals.
charge = n_elec
# old_E = energy(v, rho)
n_steps = 1000000
max_step = 0.0001

energies = []


def simpleOpt(rho):
    N = 100000
    for i in range(N):
        trial = np.sqrt(rho)
        pot = funcDeriv(rho)
        mu = chemicalPotential(rho)
        next_func = trial - 2 * max_step * trial * (pot - mu)

        rho = np.copy(next_func**2)
        rho = n_elec * rho / rho.sum() / dx
        E = energy(rho)
        energies.append(E)
        if len(energies) > 1:
            dE = np.abs(energies[-1] - energies[-2])

            if dE < 1e-6:
                print('Converged.')
                print(i, E, np.sum(pot - mu))
                break
        if i % 1000 == 0:
            print(i, E, np.sum(pot - mu))
    # np.savetxt('true_rho.dat', rho)
    # np.savetxt('true_pot.dat', v + hartree(rho))
    plt.scatter(x, v, label=r'$V(x)$')
    plt.scatter(x, rho, label=r'$\rho(x)$')
    plt.scatter(x, hartree(rho), label=r'$V_{H}(x)$')
    plt.legend()
    plt.savefig(f'true_density_N_{n_elec}.png')


# simpleOpt(rho)
# print(rho.sum() * dx)

# rho_gauss = norm.pdf(x)


def f(x, xp):
    z = np.sqrt((x-xp)**2)
    a = norm.pdf(x)*norm.pdf(xp)
    return a/(z + 1E-4)


yy = f(x.reshape(-1, 1), x.reshape(1, -1))
integral = simps([simps(yy_x, x) for yy_x in yy], x)
print(integral)
vh = []
for xi in x:
    z = np.sqrt((x-xi)**2 + 1E-7)
    a = norm.pdf(x)
    int_ = a/(z)
    integ_ = simps(int_, x)
    vh.append(integ_)

plt.scatter(x, norm.pdf(x), label=r'$\rho(x)$')
plt.scatter(x, np.array(vh), label=r'$V_{H}(x)$')
plt.scatter(x, np.array(vh)*norm.pdf(x),
            label=r'$V_{H}(x) \times \rho(x)$')
plt.legend()
plt.show()

# for i in range(n_steps):
#     rand_ind = np.random.randint(1024)
#     rand_step = np.random.uniform(-max_step, max_step)

#     old_val = rho[rand_ind]
#     if old_val + rand_step < 0:
#         continue

#     trial_rho = np.copy(rho)
#     trial_rho[rand_ind] += rand_step
#     trial_E = energy(v, trial_rho)

#     if trial_E < old_E:
#         rho = trial_rho
#         old_E = trial_E
#     if i % 10000 == 0:
#         print("Step: {}, Energy: {}".format(i, old_E))
# plt.scatter(x, rho)
# plt.show()
