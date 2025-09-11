import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Network parameters
num_epoch = 10000  # Number of training epochs

print('#epoch=', num_epoch)

# Fixing random seed
seed = 1
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
def loss_function(net, t_data, u_data):

    # Data
    u_pred = net(t_data)
    loss_data = torch.mean((u_pred - u_data) ** 2)

    return loss_data

''' Data & model '''

# Data load
tu_data = torch.from_numpy(np.loadtxt('data/data_noisy.txt', dtype='float32'))
t_data = tu_data[:,[0]]
u_data = tu_data[:,[1]]

# Model assign
net = NN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

''' Training '''

# Model update
fw = open('result/loss_1b_nn_noisy.txt', 'w')
for epoch in range(num_epoch):
    optimizer.zero_grad()
    loss = loss_function(net, t_data, u_data)
    loss.backward()
    optimizer.step()

    fw.write(str(epoch)+'\t'+str('%.4e'%loss)+'\n')
    if epoch % 1000 == 0:
        print('Epoch', epoch, 'Loss:', str('%.4e'%loss))

# Save
torch.save(net, 'result/net_1b_nn_noisy.ckpt')
loss = loss_function(net, t_data, u_data)
fw.write(str(num_epoch)+'\t'+str('%.4e'%loss)+'\n')
fw.close()
print('Epoch', num_epoch, 'Loss:', str('%.4e'%loss))

''' Inference '''

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
plt.figure()
plt.plot(t_true, u_true, label='Truth')
plt.plot(t_true, u_pred, label='NN')
plt.scatter(t_data, u_data, c='k', label='Data')
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('NN: Noisy Data')
plt.savefig('figure/1b_nn_noisy.png')
plt.close()

# Learning curve
loss = np.loadtxt('result/loss_1b_nn_noisy.txt').T
plt.figure()
plt.plot(loss[0], loss[1])
plt.yscale('log')
plt.xlim(0, num_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NN: Noisy Data')
plt.savefig('figure/loss_1b_nn_noisy.png')
plt.close()
