
import torch
device = 'cuda:0'

model = torch.nn.DataParallel(torch.nn.Linear(1000,1000))

model.to(device)

while True:
    x = torch.rand((1000, 1000), device=device)
    x = model(x)
    pass
