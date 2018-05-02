import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn

x = torch.tensor([0.1], requires_grad=True)
y = torch.asin(x)
g1 = grad(y, x, create_graph=True)
g2 = grad(g1, x, create_graph=True)
g3 = grad(g2, x, create_graph=True)
print(g3)
