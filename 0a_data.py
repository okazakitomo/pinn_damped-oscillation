import torch
import numpy as np
import matplotlib.pyplot as plt

m = 1.0  # Mass
c = 0.1  # Damping coefficient
k = 1.0  # Spring constant

noise = 0.05  # Observational noise

num_data = 10  # Number of data points
num_plot = 101  # Number of plot points

print('m =', m, ', c =', c, ', k =', k)

# Fixing random seed
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Data generator
def generator(m, c, k, tmin, tmax, num_data, noise=0):
    t = np.linspace(tmin, tmax, num_data)
    gamma = c / (2 * m)
    omega = np.sqrt(k / m - gamma ** 2)
    u = np.exp(-gamma * t) * np.cos(omega * t)  # Exact solution
    u += noise * np.random.randn(num_data)
    return t, u

# Generate data
t_exact, u_exact = generator(m, c, k, 0.5, 9.5, num_data)  # Exact data
t_noisy, u_noisy = generator(m, c, k, 0.5, 9.5, num_data, noise)  # Noisy data
t_truth, u_truth = generator(m, c, k, 0, 10, num_plot)  # Ground truth

# Save data
fe = open('data/data_exact.txt', 'w')  # Exact data (t, u)
fn = open('data/data_noisy.txt', 'w')  # Noisy data (t, u)
ft = open('data/data_truth.txt', 'w')  # Ground truth (t, u)
for i in range(num_data):
    fe.write(str('%.1f'%t_exact[i])+'\t'+str('%.4f'%u_exact[i])+'\n')
    fn.write(str('%.1f'%t_noisy[i])+'\t'+str('%.4f'%u_noisy[i])+'\n')
for i in range(num_plot):
    ft.write(str('%.1f'%t_truth[i])+'\t'+str('%.4f'%u_truth[i])+'\n')
fe.close()
fn.close()
ft.close()

# Plot
plt.figure()
plt.plot(t_truth, u_truth, label='Truth')
plt.scatter(t_exact, u_exact, c='k', label='Data')
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('Exact Data')
plt.savefig('figure/0_data_exact.png')
plt.close()

plt.figure()
plt.plot(t_truth, u_truth, label='Truth')
plt.scatter(t_noisy, u_noisy, c='k', label='Data')
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('Noisy Data')
plt.savefig('figure/0_data_noisy.png')
plt.close()

plt.figure()
plt.plot(t_truth, u_truth)
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Ground Truth')
plt.savefig('figure/0_data_truth.png')
plt.close()
