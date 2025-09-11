import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

m = 1.0  # Mass
k = 1.0  # Spring constant

nc = 101 # Number of damping coefficient
nt = 101 # Number of time

c_list = np.arange(nc) / (nc - 1)  # Damping coefficient in [0, 1]

# Colormap
norm = Normalize(vmin=0.0, vmax=1.0)
cmap = cm.jet

# Data generator
def generator(m, c, k, tmin, tmax, num_data, noise=0):
    t = np.linspace(tmin, tmax, num_data)
    omega_0 = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    omega_d = omega_0 * np.sqrt(1 - zeta**2)
    u = np.exp(-zeta * omega_0 * t) * np.cos(omega_d * t)  # Exact solution
    u += noise * np.random.randn(num_data)
    return t, u

# Generate data, Save data, Plot
plt.figure()
ft = open('data/data_truth_c.txt', 'w')  # Ground truth (c, t, u)

for c in c_list:
    t_truth, u_truth = generator(m, c, k, 0, 10, nt)  # Ground truth
    if c * (2 * c - 1) * (c - 1) == 0:
        plt.plot(t_truth, u_truth, c=cmap(norm(c)), label='c='+str('%.1f'%c))
    else:
        plt.plot(t_truth, u_truth, c=cmap(norm(c)))
    for i in range(nt):
        ft.write(str('%.2f'%c)+'\t'+str('%.1f'%t_truth[i])+'\t'+str('%.4f'%u_truth[i])+'\n')

ft.close()
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Ground Truth')
plt.legend()
plt.savefig('figure/0_data_truth_c.png')
plt.close()

truth = np.loadtxt('data/data_truth_c.txt').T
plt.figure()
plt.scatter(truth[1], truth[0], marker='s', s=5, c=truth[2], cmap='seismic', vmin=-1, vmax=1)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.xlabel('t')
plt.ylabel('c')
plt.title('Ground Truth')
plt.colorbar()
plt.savefig('figure/0_simul_truth.png')
plt.close()
