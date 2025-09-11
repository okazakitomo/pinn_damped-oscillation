import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
m = 1.0  # Mass
c = 0.1  # Damping coefficient
k = 1.0  # Spring constant

# Initial conditions
u0 = 1.0  # Initial displacement
v0 = 0.0  # Initial velocity

# Analysis range
tmin = 0
tmax = 10

# Network parameters
num_coll = 101  # Number of collocation points
num_epoch = 10000  # Number of training epochs

print('m =', m, ', c =', c, ', k =', k)
print('u0 =', u0, ', v0 =', v0)
print('tmin =', tmin, ', tmax =', tmax)
print('#coll =', num_coll, ', #epoch=', num_epoch)

# Fixing random seed
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

''' Class & function '''

# Neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Loss function
def loss_function(net, t_coll, m, c, k, u0, v0):

    # ODE
    u = net(t_coll)  # u(t)
    u_t = torch.autograd.grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # du/dt
    u_tt = torch.autograd.grad(u_t, t_coll, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]  # d2u/dt2
    r = m * u_tt + c * u_t + k * u  # ODE residual
    loss_ode = torch.mean(r ** 2)

    # IC
    t0 = torch.zeros((1, 1), requires_grad=True)  # t = 0
    u0_pred = net(t0) # u(0)
    v0_pred = torch.autograd.grad(u0_pred, t0, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]  # du/dt(0)
    loss_ic = torch.mean((u0_pred - u0) ** 2 + (v0_pred - v0) ** 2)

    # Total
    loss = loss_ode + loss_ic
    return loss, loss_ode, loss_ic

# Collocation points
def uniform_grid(tmin, tmax, num_coll):
    return torch.linspace(tmin, tmax, num_coll, requires_grad=True).view(-1, 1)

''' Model '''

# Model assign
net = NN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
t_coll = uniform_grid(tmin, tmax, num_coll)

''' Training '''

# Model update
fw = open('result/loss_2_forward.txt', 'w')
for epoch in range(num_epoch):
    optimizer.zero_grad()
    loss, loss_ode, loss_ic = loss_function(net, t_coll, m, c, k, u0, v0)
    loss.backward()
    optimizer.step()

    fw.write(str(epoch)+'\t'+str('%.4e'%loss)+'\t'+str('%.4e'%loss_ode)+'\t'+str('%.4e'%loss_ic)+'\n')
    if epoch % 1000 == 0:
        print('Epoch', epoch, 'Loss:', str('%.4e'%loss), 'ODE:', str('%.4e'%loss_ode), 'IC:', str('%.4e'%loss_ic))

# Save
torch.save(net, 'result/net_2_forward.ckpt')
loss, loss_ode, loss_ic = loss_function(net, t_coll, m, c, k, u0, v0)
fw.write(str(num_epoch)+'\t'+str('%.4e'%loss)+'\t'+str('%.4e'%loss_ode)+'\t'+str('%.4e'%loss_ic)+'\n')
fw.close()
print('Epoch', num_epoch, 'Loss:', str('%.4e'%loss), 'ODE:', str('%.4e'%loss_ode), 'IC:', str('%.4e'%loss_ic))

''' Evaluation '''

# Exact solution
tu_true = torch.from_numpy(np.loadtxt('data/data_truth.txt', dtype='float32'))
t_true = tu_true[:,[0]]
u_true = tu_true[:,[1]]

# Model prediction
net.eval()
with torch.no_grad():
    u_pred = net(t_true)

# Error
nrm = torch.sqrt(torch.mean(u_true ** 2))
abs_err = torch.sqrt(torch.mean((u_pred - u_true) ** 2))
rel_err = abs_err / nrm
print('Ablolute error:', '%.4f'%abs_err)
print('Relative error:', '%.4f'%rel_err)

''' Figure '''

# Solution
plt.plot(t_true, u_true, label='Truth')
plt.plot(t_true, u_pred, label='PINN')
plt.xlim(tmin, tmax)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('Forward')
plt.savefig('figure/2_forward.png')
plt.close()

# Learning curve
loss = np.loadtxt('result/loss_2_forward.txt').T
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
plt.savefig('figure/loss_2_forward.png')
plt.close()
