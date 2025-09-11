import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
m = 1.0  # Mass
k = 1.0  # Spring constant

# Initial conditions
u0 = 1.0  # Initial displacement
v0 = 0.0  # Initial velocity

# Analysis range
cmin = 0
cmax = 1
tmin = 0
tmax = 10

# Network parameters
num_coll_c = 21 # Number of collocation points for c
num_coll_t = 51 # Number of collocation points for t
num_epoch = 10000 # Number of training epochs

print('m =', m, ', k =', k)
print('u0 =', u0, ', v0 =', v0)
print('cmin =', cmin, ', cmax =', cmax)
print('tmin =', tmin, ', tmax =', tmax)
print('#coll c =', num_coll_c, '#coll t =', num_coll_t, ', #epoch=', num_epoch)

# Fixing random seed
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

''' Class & function '''

# Neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 20)  # Input (c, t)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# Loss function
def loss_function(net, ct_coll, ct_init, m, k, u0, v0):
    # ODE
    u = net(ct_coll).reshape(-1)  # u(c,t)
    u_ct = torch.autograd.grad(u, ct_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # (du/dc, du/dt)
    u_t = u_ct[:,1]  # du/dt
    u_tct = torch.autograd.grad(u_t, ct_coll, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]  # (d2u/dcdt, d2u/dt2)
    u_tt = u_tct[:,1]  # d2u/dt2
    r = m * u_tt + ct_coll[:,0] * u_t + k * u  # ODE residual: c is an NN input
    loss_ode = torch.mean(r ** 2)

    # IC
    u0_pred = net(ct_init).reshape(-1)  # u(c,0)
    u0_ct = torch.autograd.grad(u0_pred, ct_init, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]  # (du/dc(0), du/dt(0))
    v0_pred = u0_ct[:,1]  # du/dt(0)
    loss_ic = torch.mean((u0_pred - u0) ** 2 + (v0_pred - v0) ** 2)

    loss = loss_ode + loss_ic
    return loss, loss_ode, loss_ic

# Collocation points
def uniform_grid(cmin, cmax, tmin, tmax, num_coll_c, num_coll_t):
    c_coll = torch.linspace(cmin, cmax, num_coll_c, requires_grad=True)
    ct_init = torch.cat((c_coll.reshape(-1, 1), torch.zeros(num_coll_c, 1)), 1)  # Collocation points (c, 0)
    t_coll = torch.linspace(tmin, tmax, num_coll_t, requires_grad=True)
    c_coll, t_coll = torch.meshgrid(c_coll, t_coll)
    ct_coll = torch.cat((c_coll.reshape(-1, 1), t_coll.reshape(-1, 1)), 1)  # Collocation points (c, t)
    return ct_coll, ct_init

''' Model '''

# Model assign
net = NN()
optimizer = optim.Adam(net.parameters(), lr=0.001)
ct_coll, ct_init = uniform_grid(cmin, cmax, tmin, tmax, num_coll_c, num_coll_t)

''' Training '''

# Model update
fw = open('result/loss_4_simul.txt', 'w')
for epoch in range(num_epoch):
    optimizer.zero_grad()
    loss, loss_ode, loss_ic = loss_function(net, ct_coll, ct_init, m, k, u0, v0)
    loss.backward()
    optimizer.step()

    fw.write(str(epoch)+'\t'+str('%.4e'%loss)+'\t'+str('%.4e'%loss_ode)+'\t'+str('%.4e'%loss_ic)+'\n')
    if epoch % 1000 == 0:
        print('Epoch', epoch, 'Loss:', str('%.4e'%loss), 'ODE:', str('%.4e'%loss_ode), 'IC:', str('%.4e'%loss_ic))

# Save
torch.save(net, 'result/net_4_simul.ckpt')
loss, loss_ode, loss_ic = loss_function(net, ct_coll, ct_init, m, k, u0, v0)
fw.write(str(num_epoch)+'\t'+str('%.4e'%loss)+'\t'+str('%.4e'%loss_ode)+'\t'+str('%.4e'%loss_ic)+'\n')
fw.close()
print('Epoch', num_epoch, 'Loss:', str('%.4e'%loss), 'ODE:', str('%.4e'%loss_ode), 'IC:', str('%.4e'%loss_ic))

''' Evaluation '''

# Exact solution
ctu_true = torch.from_numpy(np.loadtxt('data/data_truth_c.txt', dtype='float32'))
ct_true = ctu_true[:,[0,1]]
u_true = ctu_true[:,[2]]

# Model prediction
net.eval()
with torch.no_grad():
    u_pred = net(ct_true)

# Error
nrm = torch.sqrt(torch.mean(u_true ** 2))
abs_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2))
rel_err = abs_err / nrm
print('Ablolute error:', '%.4f'%abs_err)
print('Relative error:', '%.4f'%rel_err)

''' Figure '''

# Solution (2D)
plt.figure()
plt.scatter(ct_true[:,1], ct_true[:,0], marker='s', s=5, c=u_pred, cmap='seismic', vmin=-1, vmax=1)
plt.xlim(tmin, tmax)
plt.ylim(cmin, cmax)
plt.xlabel('t')
plt.ylabel('c')
plt.title('Simultaneous')
plt.colorbar()
plt.savefig('figure/4_simul.png')
plt.close()

# Solution (1D)
for c in (np.arange(11) / 10):
    mask = (ct_true[:,0] > c - 1e-6) * (ct_true[:,0] < c + 1e-6)
    t_mask = ct_true[mask,1]
    u_true_mask = u_true[mask]
    u_pred_mask = u_pred[mask]

    plt.figure()
    plt.plot(t_mask, u_true_mask, label='Truth')
    plt.plot(t_mask, u_pred_mask, label='PINN')
    plt.xlim(tmin, tmax)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.legend()
    plt.title('Simultaneous: c='+str('%.1f'%c))
    plt.savefig('figure/4_simul_c'+str('%.1f'%c)+'.png')
    plt.close()

# Learning curve
loss = np.loadtxt('result/loss_4_simul.txt').T
plt.figure()
plt.plot(loss[0], loss[1], label='Total')
plt.plot(loss[0], loss[2], label='ODE')
plt.plot(loss[0], loss[3], label='IC')
plt.yscale('log')
plt.xlim(0, num_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Forward')
plt.legend()
plt.savefig('figure/loss_4_simul.png')
plt.close()
