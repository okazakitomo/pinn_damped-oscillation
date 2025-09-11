import torch

# Function
def y(x):
    return x ** 2 + torch.sin(x)

# Input values
x = torch.tensor([-1., 0., 1.], requires_grad=True)

# Output values
y = y(x)

# Automatic differentiation
dy_auto = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
ddy_auto = torch.autograd.grad(dy_auto, x, grad_outputs=torch.ones_like(y))[0]

# Symbolic differentiation
def dy(x):
    return 2 * x + torch.cos(x)

def ddy(x):
    return 2 - torch.sin(x)

dy_symb = dy(x)
ddy_symb = ddy(x)

# Comparison
print('dy/dx (automatic diff):', dy_auto)
print('dy/dx (symbolic diff) :', dy_symb)

print('d2y/dx2 (automatic diff):', ddy_auto)
print('d2y/dx2 (symbolic diff) :', ddy_symb)
