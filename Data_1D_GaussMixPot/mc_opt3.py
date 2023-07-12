import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.ndimage import gaussian_filter as gf
import torch
import torch.nn as nn
np.random.seed(14)

class DNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1)
            )

    def forward(self, x):
        return self.model(x)

##### why is there no derivative term? this would also promote continuity. is continuity desired?
nodes=256# the last node does not count for integration but is included for simpler symmetrization
L = 10
x = np.linspace(-L/2, L/2, nodes)
k = 1.0
dx = x[1] - x[0]
v = - 1.0 * np.exp(-(x-0.5)**2) - 2.0 * np.exp(-(x + 1)**2)


def tf(rho):
    return np.pi **2 * rho **3 / 6.

def gradient(rho):
    deriv = np.zeros_like(rho)
    for i in range(rho.shape[0]):
        prev = rho[i - 1]
        next = rho[(i + 1) % rho.shape[0]]
        deriv[i] = (next-prev) / dx / 2.0
    return deriv


def vw(rho):
    return gradient(rho)**2 / rho / 8.

model = torch.load('tf_1d_model.torch')
# average error on training set
# avg_err = -2.0925359e-06
def kinetic(rho, alpha=1.0, beta=0.0):
    a = model(torch.from_numpy(rho.astype(np.float32).reshape(rho.shape + (1,))))
    # a -= avg_err
    # b = vw(rho)
    return alpha * a.detach().numpy().reshape(rho.shape)

def external(rho):
    return v 

def hartree(rho):
    dk = 2 * np.pi / dx
    k = (np.arange(nodes) - nodes //2) * dk
    k[nodes//2] = 1
    invk = 1 / k
    invk[nodes//2] = 0.
    rho_g = fft(rho)
    v_h = rho_g * (invk**2)
    v_h[nodes//2] = 0.
    return ifft(v_h).real


def exchangeEnergy(rho):
    A = - (3. / 4.) * (3. / np.pi)**(1. / 3.)
    return A * rho**(4. / 3.) 
def exchangePotential(rho):
    A = - (3. / np.pi)**(1. / 3.)
    return A * rho**(1. / 3.)

def correlationEnergy(rho):
    a = -np.log(np.sqrt(2. * np.pi)) - 0.75
    b = 0.359933
    return 2 * a * rho**2 + 2**(3/2.) * b * rho**(5/2.)

def energy(rho):
    edens = kinetic(rho) + external(rho) * rho + 0.5 * hartree(rho) * rho 
    return edens.sum()*dx

# SHO
# v = 0.5 * k * x**2

# soft coloumb
# v = -1 / (x**2 + 0.01**2)**0.5


# rho is normalized?
n_elec = 2
rho = 2 * np.ones_like(x) / L
# rho = np.zeros_like(x)
charge = n_elec  # charge changes during optimization and will be corrected to 1, charge from the last node flows into the middle, however last node charge is not counted in integrals.
old_E = energy(rho)
n_steps = 2000000
max_step = 0.001

###### symmetrized updates   update-mean(update)  or update -update[rand_inx] or something else
###### annealing means optimize the cooling schedule
###### I use very specific cluster updates

init_mean=0
init_std=0.00001
init_T=0.0001

conv = 1 / (8.617e-5 / 27.2)
rhos = []
for i in range(n_steps):
    
    #std=init_std/(1+np.log(1+i))
    #T=init_T/(1+np.log(1+i))
        
    std=init_std/1.000002**i
    T=init_T/1.000002**i
    
    ##optimal:
    #std=init_std/1.000001**i
    #T=init_T/1.000001**i
    
    #rand_change = np.random.normal(init_mean,std,size=nodes)  ## i can do symmetric functions around mean updates
    #rand_step = rand_change-rand_change.mean()
    
    rand_change = 1000*(rho+std*10)*(np.random.normal(init_mean,std,size=nodes))  ## i can do symmetric functions around mean updates
    # rand_change = gf(rand_change, 0.5)
    # rand_change = 0.5*(rand_change + np.flip(rand_change))
    #rand_step = rand_change-rand_change.mean() 
    rand_step = rand_change-rho/rho.mean()*rand_change.mean()    
    # rand_step = gf(rand_step, 1.0)
    proposed_rho = rho+rand_step
    proposed_rho = np.where(proposed_rho < 0, 0, proposed_rho)
    proposed_rho = charge * proposed_rho / proposed_rho.sum() / dx
        
    energy_diff=energy(proposed_rho)-energy(rho)
    if energy_diff<=0:
        rho=proposed_rho
    elif np.exp(-1/T*energy_diff) > np.random.rand():
        rho=proposed_rho
            
    if i % 10000 == 0:
        print("1 / beta: {}, std: {}, Step: {}, Energy: {}".format(T, std, i, energy(rho)))
        # charge = rho[:-1].sum()*dx
        # rho= charge * rho / rho.sum() / dx
    # if i % 100000 == 0:
        rhos.append(rho)
        # plt.plot(x, rho)
        # plt.pause(0.05)
        # plt.show()

true_rho = np.loadtxt('true_rho.dat')
true_pot = np.loadtxt('true_pot.dat')
# print('Exact energy: {}'.format(energy(v, exactSolution(rho))))
plt.plot(x, true_rho, '--')
plt.plot(x, true_pot, '--')
# for rho in rhos:
plt.plot(x, hartree(rho) + v)
plt.plot(x, rho)
np.savetxt('mc_ann_rho.dat', rho)
np.savetxt('mc_ann_pot.dat', hartree(rho) + v )
# plt.plot(x, exactSolution(rho))
plt.show()
