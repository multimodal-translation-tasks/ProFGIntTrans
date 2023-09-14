import torch
import torch.nn as nn

liner = nn.Linear(1, 4)
x = torch.rand(3,3,4)
x = torch.sum(x, dim=-1).unsqueeze(-1)
x = liner(x)
print(x)